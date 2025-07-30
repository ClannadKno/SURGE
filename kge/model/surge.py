import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import RelGraphConv
from kge import Config, Dataset
from kge.model.kge_model import RelationalScorer, KgeModel
from kge.util import similarity, KgeLoss, rat
from torch import Tensor
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from pytorch_pretrained_bert.modeling import BertEncoder, BertConfig, BertLayerNorm, BertPreTrainedModel
from functools import partial
from kge.util import sc
from kge.util import mixer2

class SubgraphGNN(nn.Module):
    def __init__(self, dim, num_relations, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            RelGraphConv(dim, dim, num_rels=num_relations, regularizer=None, self_loop=True) for _ in range(num_layers)
        ])
        self.relu = nn.ReLU()

    def forward(self, g):
        h = g.ndata['feat']
        for layer in self.layers:
            h = layer(g, h, g.edata['etype'])
            h = self.relu(h)
        return h

class SUEGEScorer(RelationalScorer):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, configuration_key)
        self.dim = self.get_option("entity_embedder.dim")
        self.max_context_size = self.get_option("max_context_size")
        self.initializer_range = self.get_option("initializer_range")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.cls, std=self.initializer_range)
        self.global_cls = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.global_cls, std=self.initializer_range)
        self.local_mask = Parameter(torch.Tensor(1, self.dim))
        torch.nn.init.normal_(self.local_mask, std=self.initializer_range)
        self.type_embeds = nn.Embedding(100, self.dim)
        torch.nn.init.normal_(self.type_embeds.weight, std=self.initializer_range)
        self.atomic_type_embeds = nn.Embedding(4, self.dim)
        torch.nn.init.normal_(self.atomic_type_embeds.weight, std=self.initializer_range)

        self.similarity = getattr(similarity, self.get_option("similarity"))(self.dim)
        self.layer_norm = BertLayerNorm(self.dim, eps=1e-12)
        self.atomic_layer_norm = BertLayerNorm(self.dim, eps=1e-12)

        self.mixer_encoder = mixer2.MLPMixer(
            num_ctx=self.max_context_size + 1,
            dim=self.dim,
            depth=self.get_option("nlayer"),
            ctx_dim=self.get_option("ff_dim"),
            dropout=self.get_option("attn_dropout")
        )

        self.glb_avg_pooling = nn.AdaptiveAvgPool1d(1)

        config = BertConfig(0, hidden_size=self.dim,
                            num_hidden_layers=self.get_option("nlayer") // 2,
                            num_attention_heads=self.get_option("nhead"),
                            intermediate_size=self.get_option("ff_dim"),
                            hidden_act=self.get_option("activation"),
                            hidden_dropout_prob=self.get_option("hidden_dropout"),
                            attention_probs_dropout_prob=self.get_option("attn_dropout"),
                            max_position_embeddings=0,
                            type_vocab_size=0,
                            initializer_range=self.initializer_range)
        self.atom_encoder = BertEncoder(config)
        self.atom_encoder.config = config
        self.atom_encoder.apply(partial(BertPreTrainedModel.init_bert_weights, self.atom_encoder))

        self.diffusion_steps = 1000
        self.beta = torch.linspace(1e-4, 0.02, self.diffusion_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.denoiser = nn.Sequential(
            nn.Linear(3 * self.dim + 1, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * self.dim)
        )

        self.gnn = SubgraphGNN(self.dim, self.dataset.num_relations() * 2).to(self.device)

    def _add_noise(self, x, t):
        alpha_bar = self.alpha_bar.to(x.device)
        alpha_bar_t = alpha_bar[t].view(-1, 1)
        epsilon = torch.randn_like(x)
        return torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon, epsilon

    def _denoise(self, x_t, t):
        t_emb = t.float().view(-1, 1) / self.diffusion_steps
        input = torch.cat([x_t, t_emb], dim=1)
        return self.denoiser(input)

    def create_subgraphs(self, ids, ctx_list, ctx_size, t_gnn=None):
        device = ids.device
        n = ids.size(0)
        g_list = []
        local_s_list = []
        num_nodes_list = []
        for i in range(n):
            s_i = ids[i]
            num_neighbors_i = ctx_size[s_i]
            neighbor_data_i = ctx_list[s_i, :num_neighbors_i, :]  # [num_neighbors_i, 3]
            neighbor_entities_i = neighbor_data_i[:, 0].long()
            relations_i = neighbor_data_i[:, 1].long()
            all_nodes_i = torch.unique(torch.cat([torch.tensor([s_i], device=device), neighbor_entities_i]))
            global_to_local = {node.item(): idx for idx, node in enumerate(all_nodes_i)}
            local_s_i = global_to_local[s_i.item()]
            local_neighbors_i = torch.tensor([global_to_local[ne.item()] for ne in neighbor_entities_i], device=device)
            src = torch.full_like(local_neighbors_i, local_s_i)
            dst = local_neighbors_i
            g_i = dgl.graph((src, dst), num_nodes=len(all_nodes_i), idtype=torch.long).to(device)
            e = self._entity_embedder().embed(all_nodes_i).to(device)
            if t_gnn is not None:
                t_i = t_gnn[i].expand(e.size(0))
                e_diffused, _ = self._add_noise(e, t_i)
                g_i.ndata['feat'] = e_diffused
            else:
                g_i.ndata['feat'] = e
            g_i.edata['etype'] = relations_i.to(device)
            g_list.append(g_i)
            local_s_list.append(local_s_i)
            num_nodes_list.append(len(all_nodes_i))
        g_batched = dgl.batch(g_list)
        offset = 0
        global_s_idx_list = []
        for i in range(n):
            global_s_idx = offset + local_s_list[i]
            global_s_idx_list.append(global_s_idx)
            offset += num_nodes_list[i]
        return g_batched, torch.tensor(global_s_idx_list, device=device)

    def _get_encoder_output(self, p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids, output_repr=False):
        n = p_emb.size(0)
        device = p_emb.device
        if self.training:
            t_gnn = torch.randint(0, self.diffusion_steps, (n,), device=device)
        else:
            t_gnn = torch.zeros(n, device=device, dtype=torch.long)
        ctx_list, ctx_size = self.dataset.index('neighbor')
        ctx_list = ctx_list.to(ids.device)
        ctx_size = ctx_size.to(ids.device)
        g_batched, global_s_idx_list = self.create_subgraphs(ids, ctx_list, ctx_size, t_gnn=t_gnn)
        out = self.gnn(g_batched)
        e_emb = out[global_s_idx_list]
        x = torch.cat([e_emb, p_emb, t_emb], dim=1)

        if self.training:
            t_diff = torch.randint(0, self.diffusion_steps, (n,), device=device)
            x_t, epsilon = self._add_noise(x, t_diff)
            epsilon_pred = self._denoise(x_t, t_diff)
            diffusion_loss = F.mse_loss(epsilon_pred, epsilon)
        else:
            diffusion_loss = 0

        ctx_list, ctx_size = self.dataset.index('neighbor')
        ctx_list = ctx_list.to(device)
        ctx_size = ctx_size.to(device)
        ctx_ids = ctx_list[ids].to(device).transpose(1, 2)
        ctx_size = ctx_size[ids].to(device)

        if self.training:
            perm_vector = sc.get_randperm_from_lengths(ctx_size, ctx_ids.size(1))
            ctx_ids = torch.gather(ctx_ids, 1, perm_vector.unsqueeze(-1).expand_as(ctx_ids))

        ctx_ids = ctx_ids[:, :self.max_context_size]
        ctx_size[ctx_size > self.max_context_size] = self.max_context_size

        entity_ids = ctx_ids[..., 0]
        relation_ids = ctx_ids[..., 1]
        time_ids = ctx_ids[..., 2]

        ctx_size = ctx_size + 2
        attention_mask = sc.get_mask_from_sequence_lengths(ctx_size, self.max_context_size + 2)
        mask_neighbor = attention_mask[:, 2:].clone()

        if self.training and not output_repr:
            gt_mask = ((entity_ids != gt_ent.view(n, 1)) | (relation_ids != gt_rel.view(n, 1))) | (
                    time_ids != gt_tim.view(n, 1))
            ctx_random_mask = (attention_mask
                               .new_ones((n, self.max_context_size))
                               .bernoulli_(1 - self.get_option("ctx_dropout")))
            attention_mask[:, 2:] = attention_mask[:, 2:] & ctx_random_mask & gt_mask

        entity_emb = self._entity_embedder().embed(entity_ids)
        relation_emb = self._relation_embedder().embed(relation_ids)
        time_emb = self._time_embedder().embed(time_ids)

        # if self.training:
        #     x_neighbor = torch.cat([entity_emb.unsqueeze(2), relation_emb.unsqueeze(2), time_emb.unsqueeze(2)],
        #                            dim=2).view(n, self.max_context_size, -1)
        #     x_neighbor_flat = x_neighbor.view(-1, x_neighbor.size(2))
        #     t_diff_neighbor = torch.randint(0, self.diffusion_steps, (n * self.max_context_size,), device=device)
        #     x_t_neighbor_flat, epsilon_neighbor_flat = self._add_noise(x_neighbor_flat, t_diff_neighbor)
        #     epsilon_pred_neighbor_flat = self._denoise(x_t_neighbor_flat, t_diff_neighbor)
        #     mask_flat = mask_neighbor.view(-1)
        #     if mask_flat.sum() > 0:
        #         diffusion_loss_neighbor = F.mse_loss(epsilon_pred_neighbor_flat[mask_flat],
        #                                              epsilon_neighbor_flat[mask_flat])
        #     else:
        #         diffusion_loss_neighbor = 0
        #     diffusion_loss_total = diffusion_loss + diffusion_loss_neighbor
        # else:
        #     diffusion_loss_total = 0

        if self.training and self.get_option("self_dropout") > 0 and self.max_context_size > 0 and not output_repr and self.get_option("add_mlm_loss"):
            self_dropout_sample = sc.get_bernoulli_mask([n], self.get_option("self_dropout"), device)
            masked_sample = sc.get_bernoulli_mask([n], self.get_option("mlm_mask"), device) & self_dropout_sample
            t_emb[masked_sample] = self.local_mask.unsqueeze(0)
            replaced_sample = sc.get_bernoulli_mask([n], self.get_option("mlm_replace"), device) & self_dropout_sample & ~masked_sample
            t_emb[replaced_sample] = self._time_embedder().embed(torch.randint(self.dataset.num_times(),
                                                                               replaced_sample.shape, dtype=torch.long, device=device))[replaced_sample].detach()

        src = torch.cat(
            [torch.stack([e_emb, p_emb, t_emb], dim=1), torch.stack([entity_emb, relation_emb, time_emb], dim=2)
            .view(n, 3 * self.max_context_size, self.dim)], dim=1)
        src = src.reshape(n, self.max_context_size + 1, 3, self.dim)
        src = src[attention_mask[:, 1:]]
        pos = self.atomic_type_embeds(torch.arange(0, 4, device=device)).unsqueeze(0).repeat(src.shape[0], 1, 1)
        src = torch.cat([self.cls.expand(src.size(0), 1, self.dim), src], dim=1) + pos
        src = F.dropout(src, p=self.get_option("output_dropout"), training=self.training and not output_repr)
        src = self.atomic_layer_norm(src)
        out = self.atom_encoder(src,
                                self.convert_mask(src.new_ones(src.size(0), src.size(1), dtype=torch.long)),
                                output_all_encoded_layers=False)[-1][:, 0]
        src = out.new_zeros(n, self.max_context_size + 1, self.dim)
        src[attention_mask[:, 1:]] = out
        if self.max_context_size == 0:
            return src[:, 0], 0
        src = F.dropout(src, p=self.get_option("hidden_dropout"), training=self.training)
        src = self.layer_norm(src)
        out = self.mixer_encoder(src)
        out_pooling = self.glb_avg_pooling(out.view(n, self.dim, -1)).view(n, self.dim)


        if self.training and self.get_option("add_mlm_loss") and self.get_option("self_dropout") > 0.0 and self_dropout_sample.sum() > 0:
            all_time_emb = self._time_embedder().embed_all()
            all_time_emb = F.dropout(all_time_emb, p=self.get_option("output_dropout"), training=self.training)
            source_scores = self.similarity(out_pooling, all_time_emb, False).view(n, -1)
            self_pred_loss = F.cross_entropy(
                source_scores[self_dropout_sample], t_ids[self_dropout_sample], reduction='mean')
            return out_pooling, e_emb, 2 * self_pred_loss + diffusion_loss
        else:
            return out_pooling, e_emb, 0

    def convert_mask_rat(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).repeat(1, attention_mask.size(1), 1)
        return attention_mask

    def convert_mask(self, attention_mask):
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask.float()) * -10000.0
        return attention_mask

    def _scoring(self, s_emb, p_emb, o_emb, t_emb, is_pairwise, ids, gt_ent, gt_rel, gt_tim, t_ids):
        encoder_output, e_emb, self_pred_loss = self._get_encoder_output(p_emb, t_emb, ids, gt_ent, gt_rel, gt_tim, t_ids)
        o_emb = F.dropout(o_emb, p=self.get_option("output_dropout"), training=self.training)
        g_scores = self.similarity(e_emb, o_emb, is_pairwise).view(p_emb.size(0), -1)
        target_scores = self.similarity(encoder_output, o_emb, is_pairwise).view(p_emb.size(0), -1)
        if self.training:
            return target_scores, g_scores, self_pred_loss
        else:
            return target_scores, g_scores

    def score_emb(self, s_emb, p_emb, o_emb, t_emb, combine: str, s, o, gt_ent=None, gt_rel=None, gt_tim=None, t_ids=None):
        if combine == 'spoo' or combine == 'sp_' or combine == 'spo':
            out = self._scoring(s_emb, p_emb, o_emb, t_emb, combine.startswith('spo'), s, gt_ent, gt_rel, gt_tim, t_ids)
        elif combine == 'spos' or combine == '_po':
            out = self._scoring(o_emb, p_emb, s_emb, t_emb, combine.startswith('spo'), o, gt_ent, gt_rel, gt_tim, t_ids)
        else:
            raise Exception("Combine {} is not supported in SUEGE's score function".format(combine))
        return out

class SUEGE(KgeModel):
    def __init__(self, config: Config, dataset: Dataset, configuration_key=None):
        super().__init__(config, dataset, SUEGEScorer, configuration_key=configuration_key)
        self.loss = KgeLoss.create(config)

    def forward(self, fn_name, *args, **kwargs):
        self._scorer._entity_embedder = self.get_s_embedder
        self._scorer._relation_embedder = self.get_p_embedder
        self._scorer._time_embedder = self.get_t_embedder
        scores = getattr(self, fn_name)(*args, **kwargs)
        del self._scorer._entity_embedder
        del self._scorer._relation_embedder
        del self._scorer._time_embedder
        if fn_name == 'get_hitter_repr':
            return scores
        if self.training:
            self_loss_w = self.get_option("self_dropout")
            self_loss_w = self_loss_w / (1 + self_loss_w)
            return self.loss(scores[0], kwargs["gt_ent"]) + self.loss(scores[1], kwargs["gt_ent"]) + self_loss_w * scores[2] * scores[0].size(0)
        else:
            return scores

    def get_hitter_repr(self, s, p):
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        return self._scorer._get_encoder_output(p_emb, s, None, None, output_repr=True)

    def score_spo(self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, direction=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        if direction:
            if direction == 's':
                p_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
            else:
                p_emb = self.get_p_embedder().embed(p)
            return self._scorer.score_emb(s_emb, p_emb, o_emb, t_emb, "spo" + direction, s, o)[0].view(-1)
        else:
            raise Exception("The SUEGE model cannot compute undirected spo scores.")

    def score_sp(self, s: Tensor, p: Tensor, t: Tensor, o: Tensor = None, gt_ent=None, gt_rel=None, gt_tim=None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        t_emb = self.get_t_embedder().embed(t)
        if o is None:
            o_emb = self.get_o_embedder().embed_all()
        else:
            o_emb = self.get_o_embedder().embed(o)
        return self._scorer.score_emb(s_emb, p_emb, o_emb, t_emb, "sp_", s, None, gt_ent, gt_rel, gt_tim, t)

    def score_po(self, p: Tensor, o: Tensor, t: Tensor, s: Tensor = None, gt_ent=None, gt_rel=None, gt_tim=None) -> Tensor:
        if s is None:
            s_emb = self.get_s_embedder().embed_all()
        else:
            s_emb = self.get_s_embedder().embed(s)
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
        return self._scorer.score_emb(s_emb, p_inv_emb, o_emb, t_emb, "_po", None, o, gt_ent, gt_rel, gt_tim, t)

    def score_sp_po(self, s: Tensor, p: Tensor, o: Tensor, t: Tensor, entity_subset: Tensor = None) -> Tensor:
        s_emb = self.get_s_embedder().embed(s)
        p_emb = self.get_p_embedder().embed(p)
        p_inv_emb = self.get_p_embedder().embed(p + self.dataset.num_relations())
        o_emb = self.get_o_embedder().embed(o)
        t_emb = self.get_t_embedder().embed(t)
        if self.get_s_embedder() is self.get_o_embedder():
            if entity_subset is not None:
                all_entities = self.get_s_embedder().embed(entity_subset)
            else:
                all_entities = self.get_s_embedder().embed_all()
            sp_scores = self._scorer.score_emb(s_emb, p_emb, all_entities, t_emb, "sp_", s, None)[0]
            po_scores = self._scorer.score_emb(all_entities, p_inv_emb, o_emb, t_emb, "_po", None, o)[0]
        else:
            assert False
        return torch.cat((sp_scores, po_scores), dim=1)