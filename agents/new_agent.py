"""
NewAgent - 高性能台球AI
使用几何预分析 + 带噪声模拟评估 + 局部优化的混合策略
"""

import math
import copy
import random
import pooltool as pt
import numpy as np
from datetime import datetime

from .agent import Agent


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """分析击球结果并计算奖励分数"""
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        if first_contact_ball_id not in player_targets:
            foul_first_hit = True
    
    cue_hit_cushion = False
    target_hit_cushion = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    foul_no_rail = (len(new_pocketed) == 0 and first_contact_ball_id is not None 
                    and not cue_hit_cushion and not target_hit_cushion)
        
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        if player_targets == ['8']:
            score += 100
        else:
            score -= 200
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score


class NewAgent(Agent):
    """高性能台球AI - 几何分析 + 带噪声模拟 + 局部优化"""
    
    def __init__(self):
        super().__init__()
        # 噪声参数（与环境一致）
        self.noise_std = {
            'V0': 0.1, 'phi': 0.1, 'theta': 0.1, 'a': 0.003, 'b': 0.003
        }
        # 搜索参数 - 平衡速度和准确性
        self.num_candidates = 24
        self.num_noise_trials = 3
        self.num_local_opt = 10
        print("NewAgent (Hybrid Strategy v2) 已初始化。")
    
    # ========== 基础工具方法 ==========
    
    def _get_ball_position(self, ball):
        return ball.state.rvw[0][:2]
    
    def _is_pocketed(self, ball):
        return ball.state.s == 4
    
    def _calculate_angle(self, from_pos, to_pos):
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        return math.degrees(math.atan2(dy, dx)) % 360
    
    def _calculate_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def _get_pocket_positions(self, table):
        return {pid: table.pockets[pid].center[:2] for pid in ['lb', 'lc', 'lt', 'rb', 'rc', 'rt']}
    
    # ========== 路径和风险检测 ==========
    
    def _check_clear_path(self, balls, from_pos, to_pos, exclude_ids=None):
        """检查两点之间是否有障碍球"""
        if exclude_ids is None:
            exclude_ids = set()
        
        ball_radius = 0.028575
        path_length = self._calculate_distance(from_pos, to_pos)
        if path_length < 0.001:
            return True
            
        dx = (to_pos[0] - from_pos[0]) / path_length
        dy = (to_pos[1] - from_pos[1]) / path_length
        
        for bid, ball in balls.items():
            if bid in exclude_ids or bid == 'cue' or self._is_pocketed(ball):
                continue
            ball_pos = self._get_ball_position(ball)
            vx, vy = ball_pos[0] - from_pos[0], ball_pos[1] - from_pos[1]
            proj = vx * dx + vy * dy
            if 0 < proj < path_length:
                perp_dist = abs(vx * (-dy) + vy * dx)
                if perp_dist < 2.5 * ball_radius:
                    return False
        return True
    
    def _check_eight_ball_risk(self, balls, cue_pos, target_pos, pocket_pos, my_targets):
        """检查是否有误打黑8的风险"""
        if my_targets == ['8'] or '8' not in balls or self._is_pocketed(balls['8']):
            return False
        
        eight_pos = self._get_ball_position(balls['8'])
        ball_radius = 0.028575
        
        for (start, end) in [(cue_pos, target_pos), (target_pos, pocket_pos)]:
            path_len = self._calculate_distance(start, end)
            if path_len > 0.001:
                dx, dy = (end[0] - start[0]) / path_len, (end[1] - start[1]) / path_len
                vx, vy = eight_pos[0] - start[0], eight_pos[1] - start[1]
                proj = vx * dx + vy * dy
                if 0 < proj < path_len:
                    perp_dist = abs(vx * (-dy) + vy * dx)
                    if perp_dist < 3.5 * ball_radius:
                        return True
        return False
    
    # ========== 动作生成 ==========
    
    def _generate_aimed_action(self, cue_pos, target_pos, pocket_pos, v0=3.0):
        """生成瞄准动作"""
        ball_radius = 0.028575
        ball_to_pocket_angle = self._calculate_angle(target_pos, pocket_pos)
        hit_angle = (ball_to_pocket_angle + 180) % 360
        
        ideal_hit_x = target_pos[0] + 2 * ball_radius * math.cos(math.radians(hit_angle))
        ideal_hit_y = target_pos[1] + 2 * ball_radius * math.sin(math.radians(hit_angle))
        phi = self._calculate_angle(cue_pos, [ideal_hit_x, ideal_hit_y])
        
        return {'V0': v0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
    
    def _generate_candidate_actions(self, balls, my_targets, table):
        """生成候选动作列表"""
        actions = []
        cue_pos = self._get_ball_position(balls['cue'])
        pockets = self._get_pocket_positions(table)
        
        remaining = [bid for bid in my_targets if not self._is_pocketed(balls[bid])]
        if not remaining:
            remaining = ['8']
        
        shot_candidates = []
        for target_id in remaining:
            if self._is_pocketed(balls[target_id]):
                continue
            target_pos = self._get_ball_position(balls[target_id])
            cue_to_target = self._calculate_distance(cue_pos, target_pos)
            
            for pocket_id, pocket_pos in pockets.items():
                target_to_pocket = self._calculate_distance(target_pos, pocket_pos)
                
                path1_clear = self._check_clear_path(balls, cue_pos, target_pos, {'cue', target_id})
                path2_clear = self._check_clear_path(balls, target_pos, pocket_pos, {target_id})
                
                angle1 = self._calculate_angle(cue_pos, target_pos)
                angle2 = self._calculate_angle(target_pos, pocket_pos)
                angle_diff = abs(angle1 - angle2)
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                angle_score = 1 - angle_diff / 180
                
                eight_risk = self._check_eight_ball_risk(balls, cue_pos, target_pos, pocket_pos, my_targets)
                
                geo_score = 0
                if path1_clear and path2_clear:
                    geo_score += 100
                elif path2_clear:
                    geo_score += 50
                elif path1_clear:
                    geo_score += 25
                
                if eight_risk:
                    geo_score -= 300
                
                geo_score -= (cue_to_target + target_to_pocket) * 8
                geo_score += angle_score * 25
                
                shot_candidates.append({
                    'target_id': target_id,
                    'target_pos': target_pos,
                    'pocket_pos': pocket_pos,
                    'geo_score': geo_score,
                    'path_clear': path1_clear and path2_clear,
                    'eight_risk': eight_risk
                })
        
        shot_candidates.sort(key=lambda x: x['geo_score'], reverse=True)
        
        for candidate in shot_candidates[:8]:
            target_pos = candidate['target_pos']
            pocket_pos = candidate['pocket_pos']
            
            dist = self._calculate_distance(cue_pos, target_pos) + self._calculate_distance(target_pos, pocket_pos)
            base_v0 = min(4.5, max(1.8, 1.5 + dist * 0.7))
            
            for v0_delta in [-0.3, 0, 0.3, 0.6]:
                v0 = max(1.5, min(5.0, base_v0 + v0_delta))
                action = self._generate_aimed_action(cue_pos, target_pos, pocket_pos, v0)
                action['_geo_score'] = candidate['geo_score']
                action['_eight_risk'] = candidate['eight_risk']
                actions.append(action)
        
        if actions:
            base_action = actions[0]
            for delta_phi in [-1.5, -0.5, 0.5, 1.5]:
                perturbed = base_action.copy()
                perturbed['phi'] = (base_action['phi'] + delta_phi) % 360
                actions.append(perturbed)
        
        return actions[:self.num_candidates]
    
    # ========== 模拟评估 ==========
    
    def _simulate_shot(self, balls, table, action):
        """执行单次模拟"""
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        
        try:
            shot.cue.set_state(
                V0=action['V0'], phi=action['phi'], 
                theta=action.get('theta', 0), a=action.get('a', 0), b=action.get('b', 0)
            )
            pt.simulate(shot, inplace=True)
            return shot
        except:
            return None
    
    def _evaluate_action(self, balls, table, action, my_targets, with_noise=False):
        """评估动作"""
        if with_noise:
            noisy_action = {
                'V0': np.clip(action['V0'] + np.random.normal(0, self.noise_std['V0']), 0.5, 8.0),
                'phi': (action['phi'] + np.random.normal(0, self.noise_std['phi'])) % 360,
                'theta': np.clip(action.get('theta', 0) + np.random.normal(0, self.noise_std['theta']), 0, 90),
                'a': np.clip(action.get('a', 0) + np.random.normal(0, self.noise_std['a']), -0.5, 0.5),
                'b': np.clip(action.get('b', 0) + np.random.normal(0, self.noise_std['b']), -0.5, 0.5)
            }
        else:
            noisy_action = action
        
        shot = self._simulate_shot(balls, table, noisy_action)
        if shot is None:
            return -100
        
        last_state = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        score = analyze_shot_for_reward(shot, last_state, my_targets)
        
        if my_targets != ['8']:
            if '8' in shot.balls and shot.balls['8'].state.s == 4 and balls['8'].state.s != 4:
                return -500
        
        return score
    
    def _evaluate_with_noise_robustness(self, balls, table, action, my_targets):
        """带噪声鲁棒性的评估"""
        ideal_score = self._evaluate_action(balls, table, action, my_targets, with_noise=False)
        
        if ideal_score <= -100:
            return ideal_score, ideal_score
        
        noise_scores = []
        for _ in range(self.num_noise_trials):
            score = self._evaluate_action(balls, table, action, my_targets, with_noise=True)
            noise_scores.append(score)
        
        avg_noise_score = np.mean(noise_scores)
        combined_score = 0.4 * ideal_score + 0.6 * avg_noise_score
        
        return combined_score, ideal_score
    
    # ========== 局部优化 ==========
    
    def _local_optimize(self, balls, table, my_targets, action, current_score):
        """对动作进行局部优化"""
        best_action = action.copy()
        best_score = current_score
        
        for _ in range(self.num_local_opt):
            strategy = random.choice(['phi', 'V0', 'combined', 'fine_phi'])
            
            candidate = best_action.copy()
            if strategy == 'phi':
                candidate['phi'] = (best_action['phi'] + random.uniform(-2, 2)) % 360
            elif strategy == 'V0':
                candidate['V0'] = np.clip(best_action['V0'] + random.uniform(-0.4, 0.4), 1.5, 5.0)
            elif strategy == 'combined':
                candidate['phi'] = (best_action['phi'] + random.uniform(-1, 1)) % 360
                candidate['V0'] = np.clip(best_action['V0'] + random.uniform(-0.2, 0.2), 1.5, 5.0)
            else:
                candidate['phi'] = (best_action['phi'] + random.uniform(-0.5, 0.5)) % 360
            
            score, _ = self._evaluate_with_noise_robustness(balls, table, candidate, my_targets)
            if score > best_score:
                best_score = score
                best_action = candidate
        
        return best_action, best_score
    
    # ========== 主决策 ==========
    
    def decision(self, balls=None, my_targets=None, table=None):
        """主决策入口"""
        if balls is None:
            return self._random_action()
        
        try:
            remaining = [bid for bid in my_targets if not self._is_pocketed(balls[bid])]
            if not remaining:
                my_targets = ["8"]
            
            cue_pos = self._get_ball_position(balls['cue'])
            
            candidates = self._generate_candidate_actions(balls, my_targets, table)
            
            if not candidates:
                return self._random_action()
            
            # 第一阶段：快速筛选
            scored_actions = []
            for action in candidates:
                ideal_score = self._evaluate_action(balls, table, action, my_targets, with_noise=False)
                eight_risk = action.get('_eight_risk', False)
                
                if eight_risk and ideal_score < 50:
                    ideal_score -= 100
                
                scored_actions.append((action, ideal_score))
            
            scored_actions.sort(key=lambda x: x[1], reverse=True)
            
            # 第二阶段：噪声鲁棒性测试
            best_action = None
            best_combined_score = float('-inf')
            
            for action, ideal_score in scored_actions[:6]:
                combined_score, _ = self._evaluate_with_noise_robustness(balls, table, action, my_targets)
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_action = action
                
                if combined_score >= 40:
                    break
            
            # 第三阶段：局部优化
            if best_action and best_combined_score > -50:
                optimized_action, optimized_score = self._local_optimize(
                    balls, table, my_targets, best_action, best_combined_score
                )
                if optimized_score > best_combined_score:
                    best_action = optimized_action
                    best_combined_score = optimized_score
            
            # 安全检查
            if best_action is None or best_combined_score <= -200:
                remaining = [bid for bid in my_targets if not self._is_pocketed(balls[bid])]
                if remaining:
                    target_id = min(remaining, key=lambda bid: 
                                   self._calculate_distance(cue_pos, self._get_ball_position(balls[bid])))
                    target_pos = self._get_ball_position(balls[target_id])
                    phi = self._calculate_angle(cue_pos, target_pos)
                    return {'V0': 2.0, 'phi': phi, 'theta': 0, 'a': 0, 'b': 0}
                return self._random_action()
            
            result = {k: v for k, v in best_action.items() if not k.startswith('_')}
            return result
            
        except Exception as e:
            return self._random_action()
