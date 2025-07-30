import math
import time

import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "eval.tie_handling",
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )
        self.tie_handling = self.config.get("eval.tie_handling")
        self.is_prepared = False



        self.case_study_size = config.get("eval.case_study_size")
        self.case_study_enable = config.get("eval.case_study_enable")
        self.case_study_k = config.get("eval.case_study_k")
        self.case_study_output_file = config.get("eval.case_study_output_file")



        if self.__class__ == EntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct all indexes needed to run."""
        if self.is_prepared:
            return


        # create data and precompute indexes
        self.triples = self.dataset.split(self.config.get("eval.split"))
        for split in self.filter_splits:
            self.dataset.index(f"{split}_sp_to_o")
            self.dataset.index(f"{split}_po_to_s")
        if "test" not in self.filter_splits and self.filter_with_test:
            self.dataset.index("test_sp_to_o")
            self.dataset.index("test_po_to_s")

        # and data loader
        self.loader = torch.utils.data.DataLoader(
            self.triples,
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),
        )


        # let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):
        "Looks up true triples for each triple in the batch"
        label_coords = []
        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index(f"{split}_sp_to_o"),
                self.dataset.index(f"{split}_po_to_s"),
            )
            label_coords.append(split_label_coords)
        label_coords = torch.cat(label_coords)

        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index("test_sp_to_o"),
                self.dataset.index("test_po_to_s"),
            )
        else:
            test_label_coords = torch.zeros([0, 3], dtype=torch.long)

        batch = torch.cat(batch).reshape((-1, 4))
        return batch, label_coords, test_label_coords

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities()

        # we also filter with test data if requested
        filter_with_test = "test" not in self.filter_splits and self.filter_with_test

        # which rankings to compute (DO NOT REORDER; code assumes the order given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )

        # dictionary that maps entry of rankings to a sparse tensor containing the
        # true labels for this option
        labels_for_ranking = defaultdict(lambda: None)


        # Initiliaze dictionaries that hold the overall histogram of ranks of true
        # answers. These histograms are used to compute relevant metrics. The dictionary
        # entry with key 'all' collects the overall statistics and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # let's go
        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            batch = batch_coords[0].to(self.device)
            s, p, o, t = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]

            label_coords = batch_coords[1].to(self.device)
            if filter_with_test:
                test_label_coords = batch_coords[2].to(self.device)
                # create sparse labels tensor
                test_labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch),
                    2 * num_entities,
                    test_label_coords,
                    self.device,
                    float("Inf"),
                )
                labels_for_ranking["_filt_test"] = test_labels

            # create sparse labels tensor
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            # compute true scores beforehand, since we can't get them from a chunked
            # score table
            o_true_scores = self.model("score_spo", s, p, o, t, "o").view(-1)
            # print(f"The shape of o_true_score is {o_true_scores.shape}")
            s_true_scores = self.model("score_spo", s, p, o, t, "s").view(-1)

            # default dictionary storing rank and num_ties for each key in rankings
            # as list of len 2: [rank, num_ties]
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                ]
            )

            # calculate scores in chunks to not have the complete score matrix in memory
            # a chunk here represents a range of entity_values to score against
            if self.config.get("eval.chunk_size") > -1:
                chunk_size = self.config.get("eval.chunk_size")
            else:
                chunk_size = self.dataset.num_entities()

            # process chunk by chunk
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

                # compute scores of chunk
                if chunk_size == self.dataset.num_entities():
                    scores = self.model("score_sp_po", s, p, o, t, None)
                else:
                    scores = self.model("score_sp_po",
                        s, p, o, t, torch.arange(chunk_start, chunk_end).to(self.device)
                    )
                scores_sp = scores[:, : chunk_end - chunk_start]
                scores_po = scores[:, chunk_end - chunk_start :]


                # replace the precomputed true_scores with the ones occurring in the
                # scores matrix to avoid floating point issues
                s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
                o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
                o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
                s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()

                scores_sp[o_in_chunk_mask, o_in_chunk] = o_true_scores[o_in_chunk_mask]
                scores_po[s_in_chunk_mask, s_in_chunk] = s_true_scores[s_in_chunk_mask]



                # now compute the rankings (assumes order: None, _filt, _filt_test)
                for ranking in rankings:
                    if labels_for_ranking[ranking] is None:
                        labels_chunk = None
                    else:
                        # densify the needed part of the sparse labels tensor
                        labels_chunk = self._densify_chunk_of_labels(
                            labels_for_ranking[ranking], chunk_start, chunk_end
                        )

                        # remove current example from labels
                        labels_chunk[o_in_chunk_mask, o_in_chunk] = 0
                        labels_chunk[
                            s_in_chunk_mask, s_in_chunk + (chunk_end - chunk_start)
                        ] = 0

                    # compute partial ranking and filter the scores (sets scores of true
                    # labels to infinity)
                    (
                        s_rank_chunk,
                        s_num_ties_chunk,
                        o_rank_chunk,
                        o_num_ties_chunk,
                        scores_sp_filt,
                        scores_po_filt,
                    ) = self._filter_and_rank(
                        scores_sp, scores_po, labels_chunk, o_true_scores, s_true_scores
                    )

                    # from now on, use filtered scores
                    scores_sp = scores_sp_filt
                    scores_po = scores_po_filt

                    # update rankings
                    ranks_and_ties_for_ranking["s" + ranking][0] += s_rank_chunk
                    ranks_and_ties_for_ranking["s" + ranking][1] += s_num_ties_chunk
                    ranks_and_ties_for_ranking["o" + ranking][0] += o_rank_chunk
                    ranks_and_ties_for_ranking["o" + ranking][1] += o_num_ties_chunk

                # we are done with the chunk

            # We are done with all chunks; calculate final ranks from counts
            s_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["s_raw"][0],
                ranks_and_ties_for_ranking["s_raw"][1],
            )
            o_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["o_raw"][0],
                ranks_and_ties_for_ranking["o_raw"][1],
            )
            s_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["s_filt"][0],
                ranks_and_ties_for_ranking["s_filt"][1],
            )
            o_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["o_filt"][0],
                ranks_and_ties_for_ranking["o_filt"][1],
            )

            # Update the histograms of of raw ranks and filtered ranks
            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:
                f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
                f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)

            # and the same for filtered_with_test ranks
            if filter_with_test:
                batch_hists_filt_test = dict()
                s_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["s_filt_test"][0],
                    ranks_and_ties_for_ranking["s_filt_test"][1],
                )
                o_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["o_filt_test"][0],
                    ranks_and_ties_for_ranking["o_filt_test"][1],
                )
                for f in self.hist_hooks:
                    f(
                        batch_hists_filt_test,
                        s,
                        p,
                        o,
                        s_ranks_filt_test,
                        o_ranks_filt_test,
                        job=self,
                    )

            # optionally: trace ranks of each example
            if self.trace_examples:
                entry = {
                    "type": "entity_ranking",
                    "scope": "example",
                    "split": self.eval_split,
                    "filter_splits": self.filter_splits,
                    "size": len(batch),
                    "batches": len(self.loader),
                    "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"], entry["t"]= (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                        t[i].item(),
                    )
                    if filter_with_test:
                        entry["rank_filtered_with_test"] = (
                            o_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        event="example_rank",
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        **entry,
                    )
                    if filter_with_test:
                        entry["rank_filtered_with_test"] = (
                            s_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        event="example_rank",
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        **entry,
                    )


            # Compute the batch metrics for the full histogram (key "all")
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )

            # optionally: trace batch metrics
            if self.trace_batch:
                self.trace(
                    event="batch_completed",
                    type="entity_ranking",
                    scope="batch",
                    split=self.eval_split,
                    filter_splits=self.filter_splits,
                    epoch=self.epoch,
                    batch=batch_number,
                    size=len(batch),
                    batches=len(self.loader),
                    **metrics,
                )

            # output batch information to console
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f})"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                    metrics["mean_reciprocal_rank"],
                    metrics["mean_reciprocal_rank_filtered"],
                    metrics["hits_at_1"],
                    metrics["hits_at_1_filtered"],
                    self.hits_at_k_s[-1],
                    metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                    metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                ),
                end="",
                flush=True,
            )

            # merge batch histograms into global histograms
            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )
        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
            epoch_time=epoch_time,
            event="eval_completed",
            **metrics,
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in trace_entry:
            trace_entry[metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **trace_entry),
            )

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()
        self.config.log("Finished evaluating on " + self.eval_split + " split.")

        for f in self.post_valid_hooks:
            f(self, trace_entry)



        # 执行案例研究
        if self.case_study_enable:
            self._perform_case_study()


        return trace_entry

    def _perform_case_study(self):
        """执行案例研究并写入结果文件"""
        self.config.log(f"开始案例研究，抽样{self.case_study_size}个样本...")

        # 获取测试数据
        test_triples = self.dataset.split(self.eval_split)
        num_triples = len(test_triples)

        # 随机选择样本 (移动到CPU处理)
        selected_indices = torch.randperm(num_triples, device='cpu')[:self.case_study_size]
        selected_triples = [test_triples[i].cpu().tolist() for i in selected_indices]

        with open(self.case_study_output_file, "w") as f:
            for idx, triple in enumerate(selected_triples):
                s, p, o, t = triple
                # 批量转换ID到字符串（使用Tensor输入）
                s_name = self.dataset.entity_strings(torch.tensor([s]))[0]
                p_name = self.dataset.relation_strings(torch.tensor([p]))[0]
                o_name = self.dataset.entity_strings(torch.tensor([o]))[0]

                # 处理时间字符串（条件判断）
                if hasattr(self.dataset, 'time_strings'):
                    t_str = self.dataset.time_strings(torch.tensor([t]))[0]
                else:
                    t_str = str(t)

                # 写入真实四元组
                f.write(f"Case {idx + 1}\n")
                f.write(f"True Triple: [S] {s_name} [P] {p_name} [O] {o_name} [T] {t_str}\n")

                # 预测top-K subjects和objects
                top_objects = self._get_top_k_predictions(s, p, o, t, "s")
                top_subjects = self._get_top_k_predictions(s, p, o, t, "o")

                # 批量转换实体ID到名称
                sub_ids = torch.tensor([ent_id for ent_id, _ in top_subjects], device='cpu')
                sub_names = self.dataset.entity_strings(sub_ids)
                obj_ids = torch.tensor([ent_id for ent_id, _ in top_objects], device='cpu')
                obj_names = self.dataset.entity_strings(obj_ids)

                # 写入预测结果
                f.write("Top Predicted Subjects:\n")
                for i, (ent_id, score) in enumerate(top_subjects):
                    f.write(f"{i + 1}. {sub_names[i]} (score: {score:.3f})\n")

                f.write("\nTop Predicted Objects:\n")
                for i, (ent_id, score) in enumerate(top_objects):
                    f.write(f"{i + 1}. {obj_names[i]} (score: {score:.3f})\n")

                f.write("\n" + "=" * 80 + "\n\n")


    def _get_top_k_predictions(self, s, p, o, t, position: str):
        """获取指定位置的top-K预测实体（优化批量转换）"""
        device = self.device
        chunk_size = self.config.get("eval.chunk_size") if self.config.get(
            "eval.chunk_size") > 0 else self.dataset.num_entities()
        num_entities = self.dataset.num_entities()

        # 将输入转为tensor（确保在正确设备）
        s_tensor = torch.tensor([s], device=device)
        p_tensor = torch.tensor([p], device=device)
        o_tensor = torch.tensor([o], device=device)
        t_tensor = torch.tensor([t], device=device)

        # 分块计算所有实体的得分
        all_scores = []
        for chunk_start in range(0, num_entities, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_entities)
            entities = torch.arange(chunk_start, chunk_end, device=device)
            # 使用 "score_sp_po" 替代 "score_spo"
            chunk_scores = self.model(
                "score_sp_po",
                s_tensor,
                p_tensor,
                o_tensor,
                t_tensor,
                entities
            )
            # 根据位置提取对应得分
            if position == "s":
                chunk_scores = chunk_scores[:, :chunk_end - chunk_start]  # 主体得分
            else:  # position == "o"
                chunk_scores = chunk_scores[:, chunk_end - chunk_start:]  # 客体得分
            all_scores.append(chunk_scores.cpu())

        # 正确合并得分（沿dim=1拼接并移除batch维度）
        all_scores = torch.cat(all_scores, dim=1).squeeze(0)
        print(f"The shape of all_scores: {all_scores.shape}")  # 调试输出
        print(f"all_scores: {all_scores}")  # 调试输出


        # 确保k不超过实体数量
        k = min(self.case_study_k, num_entities)
        print(f"case_study_k: {self.case_study_k}, num_entities: {num_entities}, k: {k}")
        top_scores, top_indices = torch.topk(all_scores, k=k)

        # 转换为(id, score)元组列表
        return list(zip(
            top_indices.cpu().numpy().tolist(),
            top_scores.cpu().numpy().tolist()
        ))


    def _densify_chunk_of_labels(
        self, labels: torch.Tensor, chunk_start: int, chunk_end: int
    ) -> torch.Tensor:
        """Creates a dense chunk of a sparse label tensor.

        A chunk here is a range of entity values with 'chunk_start' being the lower
        bound and 'chunk_end' the upper bound.

        The resulting tensor contains the labels for the sp chunk and the po chunk.

        :param labels: sparse tensor containing the labels corresponding to the batch
        for sp and po

        :param chunk_start: int start index of the chunk

        :param chunk_end: int end index of the chunk

        :return: batch_size x chunk_size*2 dense tensor with labels for the sp chunk and
        the po chunk.

        """
        num_entities = self.dataset.num_entities()
        indices = labels._indices()
        mask_sp = (chunk_start <= indices[1, :]) & (indices[1, :] < chunk_end)
        mask_po = ((chunk_start + num_entities) <= indices[1, :]) & (
            indices[1, :] < (chunk_end + num_entities)
        )
        indices_sp_chunk = indices[:, mask_sp]
        indices_sp_chunk[1, :] = indices_sp_chunk[1, :] - chunk_start
        indices_po_chunk = indices[:, mask_po]
        indices_po_chunk[1, :] = (
            indices_po_chunk[1, :] - num_entities - chunk_start * 2 + chunk_end
        )
        indices_chunk = torch.cat((indices_sp_chunk, indices_po_chunk), dim=1)
        dense_labels = torch.sparse.LongTensor(
            indices_chunk,
            labels._values()[mask_sp | mask_po],
            torch.Size([labels.size()[0], (chunk_end - chunk_start) * 2]),
        ).to_dense()
        return dense_labels

    def _filter_and_rank(
        self,
        scores_sp: torch.Tensor,
        scores_po: torch.Tensor,
        labels: torch.Tensor,
        o_true_scores: torch.Tensor,
        s_true_scores: torch.Tensor,
    ):
        """Filters the current examples with the given labels and returns counts rank and
num_ties for each true score.

        :param scores_sp: batch_size x chunk_size tensor of scores

        :param scores_po: batch_size x chunk_size tensor of scores

        :param labels: batch_size x 2*chunk_size tensor of scores

        :param o_true_scores: batch_size x 1 tensor containing the scores of the actual
        objects in batch

        :param s_true_scores: batch_size x 1 tensor containing the scores of the actual
        subjects in batch

        :return: batch_size x 1 tensors rank and num_ties for s and o and filtered
        scores_sp and scores_po

        """
        chunk_size = scores_sp.shape[1]
        if labels is not None:
            # remove current example from labels
            labels_sp = labels[:, :chunk_size]
            labels_po = labels[:, chunk_size:]
            scores_sp = scores_sp - labels_sp
            scores_po = scores_po - labels_po
        o_rank, o_num_ties = self._get_ranks_and_num_ties(scores_sp, o_true_scores)
        s_rank, s_num_ties = self._get_ranks_and_num_ties(scores_po, s_true_scores)
        return s_rank, s_num_ties, o_rank, o_num_ties, scores_sp, scores_po

    @staticmethod
    def _get_ranks_and_num_ties(
        scores: torch.Tensor, true_scores: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Returns rank and number of ties of each true score in scores.

        :param scores: batch_size x entities tensor of scores

        :param true_scores: batch_size x 1 tensor containing the actual scores of the batch

        :return: batch_size x 1 tensors rank and num_ties
        """
        # process NaN values
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = true_scores.clone()
        true_scores[torch.isnan(true_scores)] = float("-Inf")

        # Determine how many scores are greater than / equal to each true answer (in its
        # corresponding row of scores)
        rank = torch.sum(scores > true_scores.view(-1, 1), dim=1, dtype=torch.long)
        num_ties = torch.sum(scores == true_scores.view(-1, 1), dim=1, dtype=torch.long)
        return rank, num_ties

    def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
        """Calculates the final rank from (minimum) rank and number of ties.

        :param rank: batch_size x 1 tensor with number of scores greater than the one of
        the true score

        :param num_ties: batch_size x tensor with number of scores equal as the one of
        the true score

        :return: batch_size x 1 tensor of ranks

        """

        if self.tie_handling == "rounded_mean_rank":
            return rank + num_ties // 2
        elif self.tie_handling == "best_rank":
            return rank
        elif self.tie_handling == "worst_rank":
            return rank + num_ties - 1
        else:
            raise NotImplementedError

    def _compute_metrics(self, rank_hist, suffix=""):
        """Computes desired matrix from rank histogram"""
        metrics = {}
        n = torch.sum(rank_hist).item()

        ranks = torch.arange(1, self.dataset.num_entities() + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            (torch.cumsum(rank_hist[: max(self.hits_at_k_s)], dim=0) / n).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics
