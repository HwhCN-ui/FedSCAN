#!/usr/bin/env python3
# Copyright (c) JLU, NODI, Huwenhan. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import time
import numpy as np
from fedscan.models.lora_utils import compute_update_norm
from fedscan.config.fedscan_config import stage1 as s1_cfg, stage2 as s2_cfg
from fedscan.config.base_config import train as train_cfg

class FedSCANAggregator:

    def __init__(self, stage_mode=1, fedscan_c=1.0, fedscan_lambda=None, fedscan_delta=None, diagnostics=False):
        self.device = train_cfg['device']
        self.lambda_m = s1_cfg['lambda_m'] if fedscan_lambda is None else float(fedscan_lambda)
        self.max_iter = s1_cfg['max_iter']
        self.tol = s1_cfg['tol'] if fedscan_delta is None else float(fedscan_delta)
        if self.lambda_m < 0:
            raise ValueError(f'fedscan_lambda must be non-negative, got {self.lambda_m}')
        if self.tol <= 0:
            raise ValueError(f'fedscan_delta must be positive, got {self.tol}')
        self.c = fedscan_c
        self.svd_mode = s2_cfg['svd_mode']
        self.spectral_k = 8
        self.benign_U = {}
        self.benign_V = {}
        self.benign_stats = {}
        self.stage1_iter = 0
        self.stage2_max_sigma = 0.0
        self.stage_mode = stage_mode
        self.diagnostics_enabled = bool(diagnostics)
        self.last_stage1_diagnostics = None
        self.last_stage2_diagnostics = None

    def _route_name(self):
        mapping = {1: 'full_fedscan', 2: 'stage1_only', 3: 'avg_plus_specclip'}
        return mapping.get(self.stage_mode, f'unknown_{self.stage_mode}')

    def init_benign_stats(self, clean_updates):
        clean_updates = [{k: v.to(self.device) for k, v in u.items()} for u in clean_updates]
        keys = list(clean_updates[0].keys())
        for key in keys:
            sigmas_all = []
            for update in clean_updates:
                param = update[key]
                full = self.svd_mode == 'full'
                U, S, Vh = torch.linalg.svd(param, full_matrices=full)
                sigmas_all.extend(S.detach().float().cpu().tolist())
            self.benign_stats[key] = float(np.median(sigmas_all)) if sigmas_all else 0.1
            avg_param = torch.mean(torch.stack([u[key] for u in clean_updates], dim=0), dim=0)
            U_b, S_b, Vh_b = torch.linalg.svd(avg_param, full_matrices=False)
            k = min(self.spectral_k, S_b.shape[0])
            self.benign_U[key] = U_b[:, :k].detach()
            self.benign_V[key] = Vh_b[:k, :].t().detach()

    def _compute_spectral_metrics(self, pre_update, post_update):
        clip_hits = []
        energy_removed = []
        overlap_Us = []
        overlap_Vs = []
        for key in pre_update.keys():
            M_pre = pre_update[key].to(self.device)
            M_post = post_update[key].to(self.device)
            if M_pre.numel() > 0:
                U0, S0, Vh0 = torch.linalg.svd(M_pre, full_matrices=False)
                cap = self.c * float(self.benign_stats.get(key, 0.1))
                if S0.numel() > 0:
                    hit_rate = float((S0 > cap).float().mean().item())
                    sumS = float(S0.sum().item()) + 1e-12
                    sumS_clip = float(torch.clamp(S0, max=cap).sum().item())
                    removed_ratio = float((sumS - sumS_clip) / sumS)
                else:
                    hit_rate, removed_ratio = (0.0, 0.0)
            else:
                hit_rate, removed_ratio = (0.0, 0.0)
            clip_hits.append(hit_rate)
            energy_removed.append(removed_ratio)
            if M_post.numel() > 0:
                U1, S1, Vh1 = torch.linalg.svd(M_post, full_matrices=False)
                k_use = min(self.spectral_k, S1.shape[0])
                if k_use > 0 and key in self.benign_U and (key in self.benign_V):
                    Ub = self.benign_U[key][:, :k_use]
                    Vb = self.benign_V[key][:, :k_use]
                    Ua = U1[:, :k_use]
                    Va = Vh1[:k_use, :].t()
                    ovU = float((Ua.t() @ Ub).norm('fro').item() ** 2 / k_use)
                    ovV = float((Va.t() @ Vb).norm('fro').item() ** 2 / k_use)
                else:
                    ovU, ovV = (0.0, 0.0)
            else:
                ovU, ovV = (0.0, 0.0)
            overlap_Us.append(ovU)
            overlap_Vs.append(ovV)
        return {'clip_hit_rate_avg': float(np.mean(clip_hits)) if clip_hits else 0.0, 'energy_removed_ratio_avg': float(np.mean(energy_removed)) if energy_removed else 0.0, 'overlap_U_avg': float(np.mean(overlap_Us)) if overlap_Us else 0.0, 'overlap_V_avg': float(np.mean(overlap_Vs)) if overlap_Vs else 0.0}

    def stage1_m_estimation(self, client_updates):
        client_updates = [{k: v.to(self.device) for k, v in u.items()} for u in client_updates]
        intermediate = self._init_avg_update(client_updates)
        if self.diagnostics_enabled:
            self.last_stage1_diagnostics = {'returned_center_weights': [1.0 / len(client_updates)] * len(client_updates), 'returned_center_iteration': 0}
        for iter in range(self.max_iter):
            inter_vec = torch.cat([intermediate[k].reshape(-1) for k in intermediate.keys()]).to(self.device)
            inter_norm = torch.norm(inter_vec) + 1e-12
            u_vecs = []
            u_norms = []
            cos_to_X = []
            for u in client_updates:
                v = torch.cat([u[k].reshape(-1) for k in u.keys()]).to(self.device)
                u_vecs.append(v)
                nv = torch.norm(v) + 1e-12
                u_norms.append(nv)
                cos_to_X.append(float(torch.dot(v, inter_vec) / (nv * inter_norm)))
            u_vecs = torch.stack(u_vecs, dim=0)
            u_norms = torch.tensor(u_norms, device=self.device)
            cos_to_X = torch.tensor(cos_to_X, device=self.device)
            deviations = []
            for u in client_updates:
                deviations.append(compute_update_norm(client_update=u, intermediate_update=intermediate))
            dev = torch.tensor(deviations, device=self.device, dtype=torch.float64)
            med = torch.median(dev)
            mad = torch.median(torch.abs(dev - med)) + 1e-12
            sigma2 = (1.4826 * mad) ** 2
            gamma = float(getattr(self, 'gamma_norm_align', 1.0))
            p = float(getattr(self, 'p_norm', 1.0))
            q = float(getattr(self, 'q_align', 1.0))
            lam = float(getattr(self, 'lambda_param', getattr(self, 'lambda_m', 0.1)))
            n_med = torch.median(u_norms)
            norm_factor = (u_norms / (n_med + 1e-12)).pow(p)
            align_factor = torch.clamp(cos_to_X, min=0.0).pow(q)
            s = dev / sigma2 * (1.0 + gamma * norm_factor * align_factor)
            weights = torch.exp(-lam * s)
            if self.diagnostics_enabled:
                raw_weights = weights.detach().cpu().tolist()
            weights = weights / (weights.sum() + 1e-12)
            if self.diagnostics_enabled:
                self.last_stage1_diagnostics.update({'last_computed_iteration': iter + 1, 'last_computed_scores': s.detach().cpu().tolist(), 'last_computed_raw_weights': raw_weights, 'last_computed_normalized_weights': weights.detach().cpu().tolist(), 'last_sigma2': float(sigma2), 'last_norm_median': float(n_med)})
            new_intermediate = {}
            for key in intermediate.keys():
                weighted_sum = torch.zeros_like(intermediate[key])
                for idx, update in enumerate(client_updates):
                    weighted_sum += weights[idx] * update[key]
                new_intermediate[key] = weighted_sum
            if self._check_convergence(intermediate, new_intermediate):
                self.stage1_iter = iter + 1
                break
            intermediate = new_intermediate
            if self.diagnostics_enabled:
                self.last_stage1_diagnostics['returned_center_weights'] = weights.detach().cpu().tolist()
                self.last_stage1_diagnostics['returned_center_iteration'] = iter + 1
        else:
            self.stage1_iter = self.max_iter
        return intermediate

    def stage2_spectral_clipping(self, intermediate_update):
        final_update = {}
        max_sigma = 0.0
        spectral_blocks = []
        for key, param in intermediate_update.items():
            param = param.to(self.device)
            full = self.svd_mode == 'full'
            U, S, Vh = torch.linalg.svd(param, full_matrices=full)
            k = S.shape[0]
            current_max = float(S.max().item()) if S.numel() > 0 else 0.0
            if current_max > max_sigma:
                max_sigma = current_max
            sigma_max = self.c * float(self.benign_stats.get(key, 0.1))
            S_clipped = torch.clamp(S, max=sigma_max)
            if self.diagnostics_enabled:
                pre_energy = float(S.double().square().sum().item())
                post_energy = float(S_clipped.double().square().sum().item())
                spectral_blocks.append({'block': key, 'shape': list(param.shape), 'cap': sigma_max, 'cap_source': 'existing_first_round_benign_spectral_statistics', 'sigma1_pre': current_max, 'sigma1_post': float(S_clipped.max().item()) if S_clipped.numel() else 0.0, 'singular_values': int(S.numel()), 'clipped_singular_values': int((S > sigma_max).sum().item()), 'frobenius_sq_pre': pre_energy, 'frobenius_sq_post': post_energy, 'frobenius_energy_removed_fraction': (pre_energy - post_energy) / pre_energy if pre_energy else 0.0})
            Sigma = torch.diag(S_clipped)
            param_clipped = U[:, :k] @ Sigma @ Vh[:k, :]
            final_update[key] = param_clipped
        self.stage2_max_sigma = max_sigma
        if self.diagnostics_enabled:
            pre_energy = sum((b['frobenius_sq_pre'] for b in spectral_blocks))
            post_energy = sum((b['frobenius_sq_post'] for b in spectral_blocks))
            count = sum((b['singular_values'] for b in spectral_blocks))
            self.last_stage2_diagnostics = {'stage2_applied': True, 'clipped_singular_value_fraction': sum((b['clipped_singular_values'] for b in spectral_blocks)) / count if count else 0.0, 'frobenius_energy_removed_fraction': (pre_energy - post_energy) / pre_energy if pre_energy else 0.0, 'blocks': spectral_blocks}
        return final_update

    def aggregate(self, client_updates, is_init_benign=False, clean_updates=None):
        if self.diagnostics_enabled:
            self.last_stage1_diagnostics = None
            self.last_stage2_diagnostics = {'stage2_applied': False}
        if is_init_benign and clean_updates:
            self.init_benign_stats(clean_updates)
        stage1_start_time = time.perf_counter()
        if self.stage_mode in [1, 2]:
            intermediate = self.stage1_m_estimation(client_updates)
        else:
            intermediate = self._init_avg_update(client_updates)
            self.stage1_iter = 0
        stage1_end_time = time.perf_counter()
        stage1_time = stage1_end_time - stage1_start_time
        stage2_start_time = time.perf_counter()
        if self.stage_mode in [1, 3]:
            final = self.stage2_spectral_clipping(intermediate)
        else:
            final = intermediate
            self.stage2_max_sigma = 0.0
        stage2_end_time = time.perf_counter()
        stage2_time = stage2_end_time - stage2_start_time
        spec = self._compute_spectral_metrics(pre_update=intermediate, post_update=final)
        fedscan_metrics = {'mode': 'fedscan', 'route': self._route_name(), 'stage1_iter': self.stage1_iter, 'stage2_max_sigma': self.stage2_max_sigma, 'stage1_time': stage1_time, 'stage2_time': stage2_time, 'total_time': stage1_time + stage2_time, 'clip_hit_rate_avg': spec['clip_hit_rate_avg'], 'energy_removed_ratio_avg': spec['energy_removed_ratio_avg'], 'overlap_U_avg': spec['overlap_U_avg'], 'overlap_V_avg': spec['overlap_V_avg']}
        return (final, fedscan_metrics)

    def _init_avg_update(self, client_updates):
        avg = {}
        for key in client_updates[0].keys():
            avg[key] = torch.mean(torch.stack([u[key].to(self.device) for u in client_updates]), dim=0)
        return avg

    def _check_convergence(self, old, new):
        for key in old.keys():
            if torch.norm(old[key] - new[key], p='fro') > self.tol:
                return False
        return True
