import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import matplotlib.font_manager as fm
import platform
from collections import defaultdict

# 한글 폰트 설정 - OS에 따라 자동 설정
def set_korean_font():
    system = platform.system()
    
    if system == 'Windows':
        font_list = ['Malgun Gothic', 'NanumGothic', 'NanumBarunGothic']
    elif system == 'Darwin':  # macOS
        font_list = ['AppleGothic', 'NanumGothic', 'NanumBarunGothic']
    else:  # Linux
        font_list = ['NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    
    for font_name in font_list:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path:
                plt.rcParams['font.family'] = font_name
                break
        except:
            continue
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

# 폰트 설정 적용
set_korean_font()

class BlueprintVisualizer:
    """
    Service Blueprint Visualizer from structured text
    Improved version to handle various edge cases and index errors
    """
    
    def __init__(self):
        self.colors = {
            'zone1': '#e3f2ff',
            'zone2': '#e6ffe6',
            'zone3': '#f3e8ff',
            'zone4': '#fff0e1',
            'touchpoint': '#bbdefb',
            'user': '#90caf9',
            'staff': '#64b5f6',
            'backstage': '#ce93d8',
            'arrow': '#1976d2'
        }
        
        self.layers = {
            'time': 63,
            'touchpoint': 54,
            'user': 43,
            'staff': 30,
            'backstage': 12  # 16에서 12로 낮춤
        }
    
    def parse_blueprint_text(self, text: str) -> dict:
        """Parse structured blueprint text into data dictionary"""
        data = {
            'service_name': '',
            'zones': [],
            'time_steps': [],
            'touchpoints': [],
            'user_actions': [],
            'staff_system': [],
            'backstage': [],
            'interactions': {}
        }
        
        lines = text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('SERVICE_NAME:'):
                data['service_name'] = line.split(':', 1)[1].strip()
            elif line == 'ZONES:':
                current_section = 'zones'
            elif line == 'TIME_STEPS:':
                current_section = 'time_steps'
            elif line == 'TOUCHPOINTS:':
                current_section = 'touchpoints'
            elif line == 'USER_ACTIONS:':
                current_section = 'user_actions'
            elif line == 'STAFF_SYSTEM:':
                current_section = 'staff_system'
            elif line == 'BACKSTAGE:':
                current_section = 'backstage'
            elif line == 'INTERACTIONS:':
                current_section = 'interactions'
            elif current_section and '|' in line:
                parts = line.split('|')
                
                try:
                    if current_section == 'zones':
                        data['zones'].append({
                            'id': parts[0],
                            'name': parts[1],
                            'start': int(parts[2]),
                            'end': int(parts[3])
                        })
                    elif current_section == 'time_steps':
                        data['time_steps'].append({
                            'step': int(parts[0]),
                            'duration': parts[1]
                        })
                    elif current_section == 'touchpoints':
                        data['touchpoints'].append({
                            'id': parts[0],
                            'name': parts[1],
                            'position': int(parts[2])
                        })
                    elif current_section == 'user_actions':
                        data['user_actions'].append({
                            'id': parts[0],
                            'name': parts[1],
                            'position': int(parts[2])
                        })
                    elif current_section == 'staff_system':
                        data['staff_system'].append({
                            'id': parts[0],
                            'name': parts[1],
                            'position': int(parts[2])
                        })
                    elif current_section == 'backstage':
                        data['backstage'].append({
                            'id': parts[0],
                            'name': parts[1],
                            'start': int(parts[2]),
                            'end': int(parts[3]),
                            'type': parts[4] if len(parts) > 4 else 'individual'
                        })
                    elif current_section == 'interactions':
                        if len(parts) >= 2:
                            interaction_type = parts[0]
                            # 쉼표로 구분된 숫자들을 파싱
                            try:
                                positions = [int(x.strip()) for x in parts[1].split(',')]
                                data['interactions'][interaction_type] = positions
                            except ValueError as e:
                                print(f"Error parsing interaction positions: {parts[1]}, Error: {e}")
                                continue
                except Exception as e:
                    print(f"Error parsing line: {line}, Error: {e}")
                    continue
        
        return data
    
    def get_interaction_offsets(self, blueprint_data):
        """각 위치별로 상호작용 타입에 따른 offset 계산"""
        position_interactions = defaultdict(list)
        
        # 모든 상호작용을 위치별로 그룹화
        for interaction_type, positions in blueprint_data.get('interactions', {}).items():
            for pos in positions:
                position_interactions[pos].append(interaction_type)
        
        # 각 위치의 상호작용별 offset 계산
        offsets = {}
        for pos, interaction_types in position_interactions.items():
            # 상호작용 타입별 우선순위와 기본 위치
            type_order = {
                'TP_UA': 0,
                'UA_SS': 1,
                'SS_BS': 2,
                'TP_SS': 3,
                'UA_BS': 4,
                'TP_BS': 5
            }
            
            sorted_types = sorted(interaction_types, key=lambda x: type_order.get(x, 10))
            
            for i, int_type in enumerate(sorted_types):
                spacing = 3.5  # 화살표 간 간격
                offset = (i - (len(sorted_types) - 1) / 2) * spacing
                offsets[(pos, int_type)] = offset
        
        return offsets
    
    def create_visualization(self, blueprint_data: dict):
        """Create visualization from parsed blueprint data with error handling"""
        
        # Calculate dynamic canvas width based on time steps
        num_steps = len(blueprint_data['time_steps'])
        x_spacing = 18
        canvas_width = max(220, (num_steps * x_spacing) + 50)
        
        # Setup canvas - 동적 크기 설정
        fig, ax = plt.subplots(figsize=(30, 18), dpi=150)
        ax.set_xlim(0, canvas_width)
        ax.set_ylim(0, 72)  # y 범위를 살짝 줄임
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Calculate positions based on actual time steps
        x_base = 25
        x_positions = [x_base + i * x_spacing for i in range(num_steps)]
        
        # Draw zones with extended color palette
        zone_colors = ['#e3f2ff', '#e6ffe6', '#f3e8ff', '#fff0e1', '#fce4ec']
        zone_edge_colors = ['blue', 'green', 'purple', 'orange', 'red']
        
        for i, zone in enumerate(blueprint_data['zones']):
            # Ensure start and end are within valid range
            start_idx = min(zone['start'], len(x_positions) - 1)
            end_idx = min(zone['end'], len(x_positions) - 1)
            
            if start_idx < len(x_positions) and end_idx < len(x_positions):
                start_x = x_positions[start_idx] - 12
                end_x = x_positions[end_idx] + 12
                width = end_x - start_x
                
                color_idx = i % len(zone_colors)
                zone_rect = FancyBboxPatch((start_x, 5), width, 58, 
                                          boxstyle="round,pad=0.1",
                                          facecolor=zone_colors[color_idx], 
                                          edgecolor=zone_edge_colors[color_idx],
                                          linewidth=3, linestyle='dashed', alpha=0.3)
                ax.add_patch(zone_rect)
                ax.text((start_x + end_x) / 2, 65, zone['name'], 
                       fontsize=18, weight='bold', ha='center', 
                       color=zone_edge_colors[color_idx])
        
        # Draw layer labels
        labels = ['TIME', 'TOUCHPOINT', 'USER\nACTIONS', 'STAFF/\nSYSTEM', 'BACKSTAGE']
        label_y = [self.layers['time'], self.layers['touchpoint'], 
                  self.layers['user'], self.layers['staff'], self.layers['backstage']]
        
        for label, y in zip(labels, label_y):
            ax.text(7, y, label, fontsize=14, weight='bold', ha='center', va='center',
                   rotation=90, bbox=dict(boxstyle="round,pad=0.4", facecolor='white'))
        
        # Draw time steps
        for i, time_step in enumerate(blueprint_data['time_steps']):
            if i < len(x_positions):
                x = x_positions[i]
                ax.text(x, self.layers['time'], time_step['duration'], 
                       fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow'))
                
                if i < len(blueprint_data['time_steps']) - 1 and i + 1 < len(x_positions):
                    arrow = FancyArrowPatch((x + 4, self.layers['time']), 
                                           (x_positions[i+1] - 4, self.layers['time']),
                                           arrowstyle='fancy,head_width=5,head_length=8', 
                                           color=self.colors['arrow'], lw=3)
                    ax.add_patch(arrow)
        
        # Draw touchpoints
        for tp in blueprint_data['touchpoints']:
            pos = tp['position']
            if pos < len(x_positions):
                x = x_positions[pos]
                box_width = max(15, len(tp['name']) * 1.2)
                rect = FancyBboxPatch((x-box_width/2, self.layers['touchpoint']-2.5), 
                                     box_width, 5,
                                     boxstyle="round,pad=0.2", 
                                     facecolor=self.colors['touchpoint'],
                                     edgecolor='navy', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, self.layers['touchpoint'], tp['name'], 
                       fontsize=13, ha='center', va='center')
        
        # Draw user actions
        prev_x = None
        prev_width = 15
        for ua in blueprint_data['user_actions']:
            pos = ua['position']
            if pos < len(x_positions):
                x = x_positions[pos]
                box_width = max(15, len(ua['name']) * 1.2)
                rect = FancyBboxPatch((x-box_width/2, self.layers['user']-2.5), 
                                     box_width, 5,
                                     boxstyle="round,pad=0.2", 
                                     facecolor=self.colors['user'],
                                     edgecolor='darkblue', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, self.layers['user'], ua['name'], 
                       fontsize=13, ha='center', va='center', weight='bold')
                
                # Draw arrow from previous action
                if prev_x is not None:
                    arrow = FancyArrowPatch((prev_x + prev_width/2, self.layers['user']), 
                                           (x - box_width/2, self.layers['user']),
                                           arrowstyle='fancy,head_width=5,head_length=8', 
                                           color=self.colors['arrow'], lw=3)
                    ax.add_patch(arrow)
                
                prev_x = x
                prev_width = box_width
        
        # Draw staff/system actions
        for ss in blueprint_data['staff_system']:
            pos = ss['position']
            if pos < len(x_positions):
                x = x_positions[pos]
                box_width = max(16, len(ss['name']) * 1.2)
                rect = FancyBboxPatch((x-box_width/2, self.layers['staff']-2.5), 
                                     box_width, 5,
                                     boxstyle="round,pad=0.2", 
                                     facecolor=self.colors['staff'],
                                     edgecolor='darkblue', linewidth=2)
                ax.add_patch(rect)
                ax.text(x, self.layers['staff'], ss['name'], 
                       fontsize=13, ha='center', va='center')
        
        # Draw backstage
        for bs in blueprint_data['backstage']:
            start_idx = min(bs['start'], len(x_positions) - 1)
            end_idx = min(bs['end'], len(x_positions) - 1)
            
            if start_idx < len(x_positions) and end_idx < len(x_positions):
                if bs['type'] == 'main':
                    # Large main system box
                    start_x = x_positions[start_idx]
                    end_x = x_positions[end_idx]
                    crm_rect = FancyBboxPatch((start_x, self.layers['backstage']-3), 
                                             end_x - start_x, 5.6,
                                             boxstyle="round,pad=0.2", 
                                             facecolor=self.colors['backstage'],
                                             edgecolor='purple', linewidth=3)
                    ax.add_patch(crm_rect)
                    ax.text((start_x + end_x) / 2, self.layers['backstage'], bs['name'], 
                           fontsize=16, ha='center', va='center', weight='bold')
                else:
                    # Individual systems
                    mid_idx = (start_idx + end_idx) // 2
                    x = x_positions[mid_idx]
                    box_width = max(17, len(bs['name']) * 1.2)
                    rect = FancyBboxPatch((x-box_width/2, self.layers['backstage']+7), 
                                         box_width, 5,
                                         boxstyle="round,pad=0.2", 
                                         facecolor='#f3e8ff',
                                         edgecolor='purple', linewidth=2)
                    ax.add_patch(rect)
                    ax.text(x, self.layers['backstage']+9.5, bs['name'], 
                           fontsize=13, ha='center', va='center')
        
        # Get offsets for all interactions
        offsets = self.get_interaction_offsets(blueprint_data)
        
        # Draw interactions with calculated offsets
        if 'TP_UA' in blueprint_data['interactions']:
            for pos in blueprint_data['interactions']['TP_UA']:
                if pos < len(x_positions):
                    x = x_positions[pos] + offsets.get((pos, 'TP_UA'), 0)
                    self._draw_bidirectional_arrow(ax, x, self.layers['touchpoint']-2.5, 
                                                 self.layers['user']+2.5, self.colors['arrow'])
        
        if 'UA_SS' in blueprint_data['interactions']:
            for pos in blueprint_data['interactions']['UA_SS']:
                if pos < len(x_positions):
                    x = x_positions[pos] + offsets.get((pos, 'UA_SS'), 0)
                    self._draw_bidirectional_arrow(ax, x, self.layers['user']-2.5, 
                                                 self.layers['staff']+2.5, '#0288d1')
        
        if 'SS_BS' in blueprint_data['interactions']:
            for pos in blueprint_data['interactions']['SS_BS']:
                if pos < len(x_positions):
                    x = x_positions[pos] + offsets.get((pos, 'SS_BS'), 0)
                    ax.plot([x, x], [self.layers['staff']-2.5, self.layers['backstage']+2.5], 
                           color='purple', lw=3, linestyle=':')
                    arrow = FancyArrowPatch((x, self.layers['backstage']+3), 
                                           (x, self.layers['backstage']+2.5),
                                           arrowstyle='fancy,head_width=5,head_length=8', 
                                           color='purple', lw=3)
                    ax.add_patch(arrow)
        
        # 추가된 상호작용 타입들
        if 'TP_SS' in blueprint_data['interactions']:
            for pos in blueprint_data['interactions']['TP_SS']:
                if pos < len(x_positions):
                    x = x_positions[pos] + offsets.get((pos, 'TP_SS'), 0)
                    # 흐린 점선으로 양방향 화살표 그리기
                    arrow1 = FancyArrowPatch((x, self.layers['touchpoint']-2.5), 
                                           (x, self.layers['staff']+2.5),
                                           arrowstyle='fancy,head_width=5,head_length=8',
                                           color='#ff9800', lw=3, linestyle='--', alpha=0.6)
                    arrow2 = FancyArrowPatch((x+1.2, self.layers['staff']+2.5), 
                                           (x+1.2, self.layers['touchpoint']-2.5),
                                           arrowstyle='fancy,head_width=5,head_length=8',
                                           color='#ff9800', lw=3, linestyle='--', alpha=0.6)
                    ax.add_patch(arrow1)
                    ax.add_patch(arrow2)
        
        if 'UA_BS' in blueprint_data['interactions']:
            for pos in blueprint_data['interactions']['UA_BS']:
                if pos < len(x_positions):
                    x = x_positions[pos] + offsets.get((pos, 'UA_BS'), 0)
                    ax.plot([x, x], [self.layers['user']-2.5, self.layers['backstage']+2.5], 
                           color='green', lw=3, linestyle=':', alpha=0.6)
                    arrow = FancyArrowPatch((x, self.layers['backstage']+3), 
                                           (x, self.layers['backstage']+2.5),
                                           arrowstyle='fancy,head_width=5,head_length=8', 
                                           color='green', lw=3, alpha=0.6)
                    ax.add_patch(arrow)
        
        if 'TP_BS' in blueprint_data['interactions']:
            for pos in blueprint_data['interactions']['TP_BS']:
                if pos < len(x_positions):
                    x = x_positions[pos] + offsets.get((pos, 'TP_BS'), 0)
                    ax.plot([x, x], [self.layers['touchpoint']-2.5, self.layers['backstage']+2.5], 
                           color='red', lw=3, linestyle=':', alpha=0.6)
                    arrow = FancyArrowPatch((x, self.layers['backstage']+3), 
                                           (x, self.layers['backstage']+2.5),
                                           arrowstyle='fancy,head_width=5,head_length=8', 
                                           color='red', lw=3, alpha=0.6)
                    ax.add_patch(arrow)
        
        plt.tight_layout()
        return fig
    
    def _draw_bidirectional_arrow(self, ax, x, y1, y2, color):
        """Draw bidirectional arrow as two separate arrows with fancy style"""
        arrow1 = FancyArrowPatch((x, y1), (x, y2),
                               arrowstyle='fancy,head_width=5,head_length=8',
                               color=color, lw=3)
        ax.add_patch(arrow1)
        arrow2 = FancyArrowPatch((x+1.2, y2), (x+1.2, y1),
                               arrowstyle='fancy,head_width=5,head_length=8',
                               color=color, lw=3)
        ax.add_patch(arrow2)
    
    def visualize_from_text(self, blueprint_text: str):
        """Main function to create visualization from text"""
        try:
            blueprint_data = self.parse_blueprint_text(blueprint_text)
            fig = self.create_visualization(blueprint_data)
            return fig
        except Exception as e:
            print(f"Error creating visualization: {e}")
            raise