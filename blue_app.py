import streamlit as st
import anthropic
from blueprint_generator import BlueprintGenerator
from blueprint_visualizer import BlueprintVisualizer
import pandas as pd
import io

# 페이지 설정
st.set_page_config(
    page_title="BlueBOM - Service Blueprint Creator",
    page_icon="🎯",
    layout="wide"
)

# 스타일 적용
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1976d2;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stDataFrame {
        font-size: 14px;
    }
    div[data-testid="stHorizontalBlock"] {
        align-items: stretch;
    }
    div[data-testid="data-editor"] {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# 헤더
st.markdown('<h1 class="main-header">BlueBOM</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">BeomAI와 함께 블루프린트를 만들어보아요</p>', unsafe_allow_html=True)

# 세션 상태 초기화
if 'blueprint_text' not in st.session_state:
    st.session_state.blueprint_text = None
if 'edit_mode' not in st.session_state:
    st.session_state.edit_mode = False
if 'blueprint_data' not in st.session_state:
    st.session_state.blueprint_data = {}
if 'has_visualization' not in st.session_state:
    st.session_state.has_visualization = False
if 'visualization_buffer' not in st.session_state:
    st.session_state.visualization_buffer = None
if 'svg_buffer' not in st.session_state:
    st.session_state.svg_buffer = None
if 'download_filename' not in st.session_state:
    st.session_state.download_filename = None

# API 키 설정
api_key = st.sidebar.text_input("Anthropic API Key", "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", 
type="password")
if not api_key:
    st.info("Please enter your Anthropic API key in the sidebar.")
    st.stop()

# 서비스명 입력
service_name = st.text_input("서비스명을 입력하세요", placeholder="예: 파리바게트에서 빵구매에 대해서 6단계")

if st.button("블루프린트 생성", disabled=not service_name):
    with st.spinner("블루프린트를 생성하고 있습니다..."):
        try:
            generator = BlueprintGenerator(api_key)
            st.session_state.blueprint_text = generator.generate_blueprint(service_name)
            st.session_state.blueprint_data = generator.parse_blueprint_text(st.session_state.blueprint_text)
            st.success("블루프린트가 생성되었습니다!")
        except Exception as e:
            st.error(f"블루프린트 생성 중 오류가 발생했습니다: {str(e)}")

# 데이터를 텍스트 형식으로 변환하는 함수
def data_to_text(data):
    text_lines = []
    text_lines.append(f"SERVICE_NAME: {data.get('service_name', '')}")
    
    text_lines.append("ZONES:")
    for zone in data.get('zones', []):
        text_lines.append(f"{zone['id']}|{zone['name']}|{zone['start']}|{zone['end']}")
    
    text_lines.append("TIME_STEPS:")
    for ts in data.get('time_steps', []):
        text_lines.append(f"{ts['step']}|{ts['duration']}")
    
    text_lines.append("TOUCHPOINTS:")
    for tp in data.get('touchpoints', []):
        text_lines.append(f"{tp['id']}|{tp['name']}|{tp['position']}")
    
    text_lines.append("USER_ACTIONS:")
    for ua in data.get('user_actions', []):
        text_lines.append(f"{ua['id']}|{ua['name']}|{ua['position']}")
    
    text_lines.append("STAFF_SYSTEM:")
    for ss in data.get('staff_system', []):
        text_lines.append(f"{ss['id']}|{ss['name']}|{ss['position']}")
    
    text_lines.append("BACKSTAGE:")
    for bs in data.get('backstage', []):
        text_lines.append(f"{bs['id']}|{bs['name']}|{bs['start']}|{bs['end']}|{bs['type']}")
    
    text_lines.append("INTERACTIONS:")
    for int_type, positions in data.get('interactions', {}).items():
        text_lines.append(f"{int_type}|{','.join(map(str, positions))}")
    
    return '\n'.join(text_lines)

# 블루프린트 표시 및 편집
if st.session_state.blueprint_text:
    st.divider()
    
    # 표시/편집 모드 토글
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("텍스트 편집" if not st.session_state.edit_mode else "적용"):
            st.session_state.edit_mode = not st.session_state.edit_mode
            if not st.session_state.edit_mode:
                # 편집 모드 종료 시 데이터 저장
                generator = BlueprintGenerator(api_key)
                st.session_state.blueprint_data = generator.parse_blueprint_text(st.session_state.blueprint_text)
                st.success("변경사항이 적용되었습니다!")
    
    # 일반 표시 모드
    if not st.session_state.edit_mode:
        data = st.session_state.blueprint_data
        
        # Service Name
        service_name_edit = st.text_input("서비스명:", value=data.get('service_name', ''), key="service_name_edit")
        if service_name_edit != data.get('service_name', ''):
            data['service_name'] = service_name_edit
            st.session_state.blueprint_text = data_to_text(data)
            # 시각화 상태 리셋
            st.session_state.has_visualization = False
            st.session_state.visualization_buffer = None
            st.session_state.svg_buffer = None
        
        # 통합 표 편집
        st.subheader("서비스 블루프린트 구성요소")
        
        # 시간 단계 수
        max_steps = len(data.get('time_steps', []))
        
        # 데이터 프레임 생성을 위한 리스트 준비
        all_data = []
        
        # ZONES 행
        zone_row = ['구역 (Zones)'] + [''] * max_steps
        for zone in data.get('zones', []):
            for i in range(zone['start'], zone['end'] + 1):
                if i < max_steps:
                    zone_row[i + 1] = f"{zone['name']}"
        all_data.append(zone_row)
        
        # TIME STEPS 행
        time_row = ['시간 단계'] + [ts['duration'] for ts in data.get('time_steps', [])]
        all_data.append(time_row)
        
        # TOUCHPOINTS 행
        tp_row = ['터치포인트'] + [''] * max_steps
        for tp in data.get('touchpoints', []):
            if tp['position'] < max_steps:
                tp_row[tp['position'] + 1] = tp['name']
        all_data.append(tp_row)
        
        # USER ACTIONS 행
        ua_row = ['사용자 활동'] + [''] * max_steps
        for ua in data.get('user_actions', []):
            if ua['position'] < max_steps:
                ua_row[ua['position'] + 1] = ua['name']
        all_data.append(ua_row)
        
        # STAFF SYSTEM 행
        ss_row = ['스태프/시스템'] + [''] * max_steps
        for ss in data.get('staff_system', []):
            if ss['position'] < max_steps:
                ss_row[ss['position'] + 1] = ss['name']
        all_data.append(ss_row)
        
        # BACKSTAGE 행
        bs_row = ['백스테이지'] + [''] * max_steps
        for bs in data.get('backstage', []):
            for i in range(bs['start'], bs['end'] + 1):
                if i < max_steps:
                    bs_row[i + 1] = bs['name']
        all_data.append(bs_row)
        
        # 열 이름 생성
        columns = ['구분'] + [f'단계 {i}' for i in range(max_steps)]
        
        # 데이터프레임 생성
        df = pd.DataFrame(all_data, columns=columns)
        
        # 편집 가능한 데이터프레임 표시
        edited_df = st.data_editor(
            df, 
            use_container_width=True, 
            height=300,
            key="blueprint_editor",
            hide_index=True,
            disabled=['구분']  # 첫 번째 열은 편집 불가
        )
        
        # 변경사항이 있으면 데이터 업데이트
        if not df.equals(edited_df):
            # 변경된 데이터를 원래 구조로 변환
            for idx, row in edited_df.iterrows():
                row_type = row['구분']
                
                if row_type == '구역 (Zones)':
                    # Zone 데이터는 복잡하므로 단순 텍스트로 표시만
                    pass
                
                elif row_type == '시간 단계':
                    for i, duration in enumerate(row[1:]):
                        if i < len(data['time_steps']) and duration:
                            data['time_steps'][i]['duration'] = duration
                
                elif row_type == '터치포인트':
                    data['touchpoints'] = []
                    for i, name in enumerate(row[1:]):
                        if name:
                            data['touchpoints'].append({
                                'id': f'TP{i}',
                                'name': name,
                                'position': i
                            })
                
                elif row_type == '사용자 활동':
                    data['user_actions'] = []
                    for i, name in enumerate(row[1:]):
                        if name:
                            data['user_actions'].append({
                                'id': f'UA{i}',
                                'name': name,
                                'position': i
                            })
                
                elif row_type == '스태프/시스템':
                    data['staff_system'] = []
                    for i, name in enumerate(row[1:]):
                        if name:
                            data['staff_system'].append({
                                'id': f'SS{i}',
                                'name': name,
                                'position': i
                            })
                
                elif row_type == '백스테이지':
                    # Backstage 데이터도 복잡하므로 단순 텍스트로 표시만
                    pass
            
            # 텍스트 업데이트
            st.session_state.blueprint_text = data_to_text(data)
            st.session_state.blueprint_data = data
            
            # 시각화 상태 리셋 - 수정 후 시각화 버튼을 다시 눌러야 함
            st.session_state.has_visualization = False
            st.session_state.visualization_buffer = None
            st.session_state.svg_buffer = None
            
            st.rerun()  # 페이지 새로고침으로 변경사항 반영
        
        # 상호작용은 별도로 표시
        if data.get('interactions'):
            st.subheader("상호작용 (Interactions)")
            
            # 상호작용 타입 설명 (확장된 버전)
            interaction_types = {
                'TP_UA': '터치포인트와 사용자 활동 간 양방향 상호작용',
                'UA_SS': '사용자 활동과 스태프/시스템 간 양방향 상호작용',
                'SS_BS': '스태프/시스템과 백스테이지 간 단방향 연결',
                'TP_SS': '터치포인트와 스태프/시스템 간 직접 상호작용',
                'UA_BS': '사용자 활동과 백스테이지 간 직접 연결',
                'TP_BS': '터치포인트와 백스테이지 간 직접 연결'
            }
            
            int_data = []
            for int_type, positions in data['interactions'].items():
                int_data.append({
                    '상호작용 타입': int_type,
                    '설명': interaction_types.get(int_type, '새로운 상호작용 타입'),
                    '발생 위치': ', '.join([f'단계 {pos}' for pos in positions])
                })
            int_df = pd.DataFrame(int_data)
            st.dataframe(int_df, use_container_width=True, hide_index=True)
    
    # 편집 모드
    else:
        st.subheader("텍스트 편집 모드")
        st.session_state.blueprint_text = st.text_area(
            "블루프린트 텍스트를 편집하세요",
            value=st.session_state.blueprint_text,
            height=400
        )
    
    # 시각화 버튼
    st.divider()
    if st.button("Visualization", type="primary"):
        with st.spinner("시각화를 생성하고 있습니다..."):
            try:
                visualizer = BlueprintVisualizer()
                fig = visualizer.visualize_from_text(st.session_state.blueprint_text)
                
                # PNG 이미지를 버퍼에 저장 - 고해상도로
                png_buf = io.BytesIO()
                fig.savefig(png_buf, format='png', dpi=300, bbox_inches='tight', facecolor='white')
                png_buf.seek(0)
                
                # SVG 이미지를 버퍼에 저장
                svg_buf = io.BytesIO()
                fig.savefig(svg_buf, format='svg', bbox_inches='tight', facecolor='white')
                svg_buf.seek(0)
                
                # 이미지와 버퍼를 세션 상태에 저장
                st.session_state.visualization_buffer = png_buf
                st.session_state.svg_buffer = svg_buf
                st.session_state.has_visualization = True
                
                # 파일명 생성
                data = st.session_state.blueprint_data
                service_name = data.get('service_name', 'blueprint').replace(' ', '_')
                st.session_state.download_filename = f"{service_name}_visualization"
                
            except Exception as e:
                st.error(f"시각화 생성 중 오류가 발생했습니다: {str(e)}")
                st.session_state.has_visualization = False
    
    # 시각화 이미지 표시 (세션 상태에서)
    if 'has_visualization' in st.session_state and st.session_state.has_visualization:
        st.image(st.session_state.visualization_buffer, caption="Service Blueprint Visualization", use_container_width=False)
        
        # 다운로드 버튼들
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="PNG 다운로드",
                data=st.session_state.visualization_buffer,
                file_name=f"{st.session_state.download_filename}.png",
                mime="image/png",
                key="download_png"
            )
        with col2:
            st.download_button(
                label="SVG 다운로드",
                data=st.session_state.svg_buffer,
                file_name=f"{st.session_state.download_filename}.svg",
                mime="image/svg+xml",
                key="download_svg"
            )

# 사이드바에 사용 가이드
with st.sidebar:
    st.header("사용 가이드")
    st.markdown("""
    1. **API Key 입력**: Anthropic API 키를 입력하세요
    2. **서비스명 입력**: 분석할 서비스명을 입력하세요
    3. **블루프린트 생성**: 버튼을 클릭하여 생성하세요
    4. **수정**: 표에서 직접 수정하거나 텍스트 편집 모드 사용
    5. **시각화**: Visualization 버튼으로 그래프를 생성하세요
    """)
    
    st.divider()
    
    st.header("상호작용 타입 설명")
    st.markdown("""
    - **TP_UA**: 터치포인트 ↔ 사용자 활동
    - **UA_SS**: 사용자 활동 ↔ 스태프/시스템
    - **SS_BS**: 스태프/시스템 → 백스테이지
    - **TP_SS**: 터치포인트 ↔ 스태프/시스템
    - **UA_BS**: 사용자 활동 → 백스테이지
    - **TP_BS**: 터치포인트 → 백스테이지
    """)
    
    st.divider()
    
    st.header("예시 서비스명")
    examples = [
        "파리바게트에서 빵구매",
        "스타벅스에서 커피 주문",
        "헬스장 이용하기",
        "온라인 쇼핑몰에서 상품 구매",
        "병원 진료 예약 및 방문"
    ]
    for example in examples:
        st.code(example)