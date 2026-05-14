import anthropic

class BlueprintGenerator:
    """
    Service Blueprint Generator using Claude API
    """
    
    def __init__(self, api_key):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.prompt_template = """당신은 서비스 블루프린트 전문가입니다. 사용자가 제공하는 서비스명에 대해 구조화된 서비스 블루프린트를 작성하세요.
서비스명: [{service_name}]

다음의 정확한 형식으로 서비스 블루프린트를 작성하세요. 반드시 아래 형식을 정확히 따라주세요:

SERVICE_NAME: [서비스 전체 이름]
ZONES:
[구역ID]|[구역명]|[시작단계]|[종료단계]
TIME_STEPS:
[단계번호]|[소요시간]
TOUCHPOINTS:
[터치포인트ID]|[터치포인트명]|[위치]
USER_ACTIONS:
[사용자활동ID]|[활동명]|[위치]
STAFF_SYSTEM:
[스태프/시스템ID]|[활동명]|[위치]
BACKSTAGE:
[백스테이지ID]|[시스템명]|[시작위치]|[종료위치]|[타입(main/individual)]
INTERACTIONS:
[상호작용타입]|[위치목록(쉼표구분)]

상호작용 타입:
- TP_UA: 터치포인트와 사용자 활동 간 양방향 상호작용
- UA_SS: 사용자 활동과 스태프/시스템 간 양방향 상호작용  
- SS_BS: 스태프/시스템과 백스테이지 간 단방향 연결
- TP_SS: 터치포인트와 스태프/시스템 간 직접 상호작용
- UA_BS: 사용자 활동과 백스테이지 간 직접 연결
- TP_BS: 터치포인트와 백스테이지 간 직접 연결

예시 형식:
SERVICE_NAME: 온라인 쇼핑몰
ZONES:
ZONE1|검색 및 선택|0|3
ZONE2|구매 및 결제|4|7
TIME_STEPS:
0|1분
1|3분
2|2분
3|30초
4|1분
5|2분
6|1분
7|30초
TOUCHPOINTS:
TP0|검색창|0
TP1|상품 목록|1
TP2|상품 상세|2
TP3|장바구니|3
TP4|결제 페이지|4
TP5|배송 정보|5
TP6|결제 확인|6
TP7|주문 완료|7
USER_ACTIONS:
UA0|검색어 입력|0
UA1|상품 둘러보기|1
UA2|상품 선택|2
UA3|장바구니 담기|3
UA4|결제 진행|4
UA5|배송지 입력|5
UA6|결제하기|6
UA7|주문 확인|7
STAFF_SYSTEM:
SS0|검색 알고리즘|0
SS1|상품 추천|1
SS2|재고 확인|2
SS3|장바구니 업데이트|3
SS4|주문 검증|4
SS5|배송 계산|5
SS6|결제 처리|6
SS7|주문 확정|7
BACKSTAGE:
BS_MAIN|전자상거래 시스템|0|7|main
BS1|재고 관리 시스템|2|3|individual
BS2|결제 시스템|4|6|individual
BS3|배송 시스템|5|7|individual
INTERACTIONS:
TP_UA|0,1,2,3,4,5,6,7
UA_SS|0,1,2,3,4,5,6,7
SS_BS|2,3,4,5,6,7
TP_SS|4,6
UA_BS|6

작성 시 주의사항:
1. 각 섹션은 정확히 위의 순서를 따라야 합니다
2. 섹션 이름은 대문자로 작성하고 콜론(:)으로 끝냅니다
3. 데이터는 파이프(|)로 구분합니다
4. INTERACTIONS의 위치 목록은 쉼표(,)로 구분합니다
5. 모든 위치(position)는 0부터 시작하는 정수여야 합니다
6. TIME_STEPS는 반드시 순차적이어야 합니다 (0, 1, 2...)
7. BACKSTAGE의 타입은 'main' 또는 'individual'만 사용합니다
8. 적절한 상호작용 타입을 선택하여 사용하세요

추가 설명이나 주석 없이 위 형식에 맞는 구조화된 데이터만 출력하세요."""
    
    def generate_blueprint(self, service_name: str) -> str:
        """Generate blueprint using Claude API"""
        message = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",  # 최신 모델명으로 변경
            max_tokens=2000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": self.prompt_template.format(service_name=service_name)
                }
            ]
        )
        
        return message.content[0].text
    
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
                        'type': parts[4]
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
        
        return data