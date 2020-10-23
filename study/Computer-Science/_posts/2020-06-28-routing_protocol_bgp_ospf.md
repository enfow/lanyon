---
layout: post
title: Routing Protocol BGP & OSPF
category_num: 510
keyword: '[IP]'
---

# Routing Protocol: BGP & OSPF

- update date : 2020.06.28
- 본 포스팅은 고려대학교 컴퓨터학과 김효곤 교수님의 2020년 1학기 Internet Protocol을 수강하고 이를 바탕으로 작성했습니다. 수업 내용 복습을 목적으로 작성하였기 때문에 내용 중 부족한 점이 있을 수 있습니다.

## 1. Introduction

한국에서 [하버드 대학 홈페이지](https://www.harvard.edu/) 에 접속한다고 하자. 하버드 대학의 홈페이지 서버는 미국에 있으므로 response를 받기 위해서는 한국에서 보낸 request 패킷이 미국의 하버드 서버까지 도달해야 한다. 이때 많은 라우터를 거쳐 가게 될 것인데, 적절한 경로를 어떻게 찾을 것인가에 대한 질문이 생긴다. 본 포스팅에서 다루는 두 가지 프로토콜 BGP와 OSPF는 내가 찾는 IP 주소를 가진 서버가 어디에 있는지, 해당 서버까지 도달하는 경로는 무엇인지 찾는 방법에 관한 것이다.

---

## 2. Two Tier Routing Archiecture

- Inter AS: BGP(Border Gateway Protocol), Policy-Oriented
- Intra AS: IGP(Interior Gateway Protocol), Mechanism-Oriented

인터넷은 **AS(Autonomous System)**라고 불리는 여러 네트워크 덩어리들로 구성된다. 구체적으로 AS는 동일한 내부 라우팅 시스템에 따라 하나의 관리자에 의해 유지되는 IP Routing Prefix들의 집합을 말하는데, 인터넷의 라우팅 프로토콜은 이러한 AS를 기준으로 AS 간 라우팅 방식과 AS 내부에서의 라우팅 방식이 다르다. 이때 BGP는 AS 간 라우팅 방식(Inter-AS)에 사용되는 프로토콜이고, OSPF는 대표적인 AS 내부에서의 라우팅 방식(Intra-AS) 중 하나라고 할수 있다.

이와 같이 라우팅 방식이 다르게 적용되는 이유는 인터넷에서 중추적인 역할을 하는 AS들이 KT, SK Broadband와 같은 **ISP(Internet Service Provider)**에 의해 관리된다는 점에 있다. 즉 AS 간의 연결 여부는 사기업인 ISP 간의 계약에 따라 좌우되며 이러한 점에서 항상 최단 경로를 택하는 것이 보장되지 않는다. 반면 AS 내부에서는 무조건 빠르게 원하는 목적지까지 패킷을 전달하는 것이 네트워크의 부하, 반응속도 측면에서 좋기 때문에 최단 경로를 선택하는 알고리즘이 사용된다. 이와 같은 차이를 두고 BGP는 **Policy-oriented**한 특성을 가지고, OSPF는 **Mechanism-oriented**한 특성을 가진다고도 표현한다.

### Stub AS, Transit AS

<img src="{{site.image_url}}/study/stub_transit_as.png" style="width:35em; display: block; margin: 0px auto;">

AS는 크게 Stub AS와 Transit AS 두 가지로 나누어 볼 수 있다. 먼저 **Stub AS**는 패킷의 출발지이자 도착지가 되는 AS이다. Stub AS에서 stub는 토막, 나머지 부분이라는 의미를 가지고 있는데 이처럼 네트워크 전체에서 끝 부분들을 말한다고 할 수 있다. 끝 부분이기 때문에 다른 네트워크로 연결되는 지점은 오로지 하나 밖에 없고, 따라서 다른 AS로부터 받은 패킷을 다른 AS에게 전달하지 않는다.

참고로 Stub AS라고 해서 반드시 하나의 연결로만 외부 AS와 연결되는 것은 아니다. 가외성을 높이기 위해 복수의 연결을 구축하기도 하는데 이를 Multi-Homing 이라고 한다.

**Transit AS**는 Stub AS와 달리 어떤 AS로부터 전달받은 패킷을 다른 AS에게 전달하는 역할을 한다. ISP의 역할이 이러한 Transit AS로 기능하며 고객(Stub AS)의 request와 response를 받아 고객이 원하는 곳에 도착할 수 있도록 다른 AS에게 넘겨주는 것이라고 할 수 있다. BGP는 Transit AS 간에 어느 AS로 전달해야 고객이 요청한 IP로 도달할 수 있는지 알려주는 데에 사용되는 프로토콜이다.

---

## 3. BGP, Border Gateway Protocol

위에서 말한 것과 같이 BGP는 AS 간에 사용되는 (거의 유일한) 라우팅 프로토콜로, 위에서 언급한 것과 같이 BGP의 역할은 특정 IP를 가진 패킷을 어떤 AS에게 전달하면 되는지 알려주는 것이다. 이를 위해 BGP는 다른 AS의 라우터와 연결 세션을 맺고 자신을 통해 도달할 수 있는 IP prefix들을 주기적으로 알린다.

<img src="{{site.image_url}}/study/bgp_aspath.png" style="width:35em; display: block; margin: 0px auto;">

위의 그림과 같이 ip prefix 180.100.100.0/24를 가지고 있는 stub AS에 연결된 transit AS(ASN 200)는 해당 IP prefix로 보내어진 패킷을 자신에게 오도록 해야 한다. 이를 위해 BGP 프로토콜을 사용하여 자신과 연결된 모든 다른 AS 들에게 180.100.100.0/24에 가기 위해서는 자신을 통해야 한다는 BGP 패킷을 전달한다. 이를 받은 다른 AS(ASN 201, 204)들은 또다시 자신과 연결된 다른 AS들에게 180.100.100.0/24로 가기 위해서는 자신을 통과해야 한다는 내용을 담아 알리게 된다. 이를 반복적으로 수행하면 모든 AS가 어떤 IP prefix에 도달하기 위한 AS 경로를 알 수 있게 된다. 

이때 자신과 다른 AS 간의 연결을 Peering이라고 하는데, 이러한 Peering은 각 AS의 이해관계에 따라 결정된다. 비슷한 규모의 AS 간에는 반대급부 없이도 Peering을 맺기도 하나 일반적으로 규모가 작은 AS는 수행할 수 있는 역할이 작으므로 규모가 큰 AS에 비용을 지불하고 Peering을 맺는 경우가 많다고 한다.

### Internal BGP, External BGP

<img src="{{site.image_url}}/study/extenal_internal_bgp.png" style="width:35em; display: block; margin: 0px auto;">

BGP가 제대로 동작하기 위해서는 AS와 Peering을 맺고 있는 다른 모든 AS에 BGP 패킷을 전달해야 한다. 하지만 Peering을 맺고 있는 BGP Speaker는 연결되어있는 AS 마다 개별적으로 존재한다. 즉 AS 200에 연결된 라우터와 AS 201에 연결된 라우터가 다를 수 있다는 것이다. 따라서 BGP 패킷을 제대로 전달하기 위해서는 AS 내부에서 브로드캐스팅을 통해 모든 BGP Speaker에 전달할 수 있어야 한다. 이와 같이 AS 내부에서 BGP 패킷을 전달하는 것을 Internal BGP라고 한다. 반대로 AS 간에 BGP 패킷을 주고 받는 것을 External BGP라고 한다.

<img src="{{site.image_url}}/study/internal_bgp.png" style="width:35em; display: block; margin: 0px auto;">

Internal BGP의 경우 위의 그림과 같이 AS 내의 다른 모든 BGP Speaker들과 모두 서로 직접 연결(**full-meshed**, directly connected)되어 있어야 한다. BGP Speaker가 아닌 다른 AS 라우터는 얼마든지 거쳐도 되지만 어떤 BGP Speaker로 가기 위해 다른 BGP Speaker를 거처야 하는 경우는 허용되지 않는다. 왜냐하면 내부 loop를 방지하기 위해 Internal BGP로 받은 패킷에 대해서는 BGP Speaker들이 더 이상 전파하지 않기 때문이다.

### BGP Packet Type

BGP 패킷은 총 4개의 타입을 가지고 있다.

|Type|Number|description|
|:------:|:---:|:---|
|**Open**|1|BGP Peering을 위한 BGP 세션 생성|
|**Update**|2|BGP 세션 업데이트|
|**Notification**|3|Error 발생|
|**Keep Alive**|4|BGP 세션 유지|

#### BGP Open

AS간 Peering이 이뤄지기 위해서는 각 AS의 BGP Speaker 간에 BGP Open 메시지를 주고 받는 작업이 필요하다. BGP Open 메시지를 통해 상대 AS가 무엇인지 인식하고, BGP 프로토콜을 위한 기본적인 정보들을 주고받으며 TCP 세션이 만들어진다. 

#### BGP KeepAlive

모든 BGP 세션은 수명이 정해져 있으며, 주기적으로 갱신해주어야 하는데 이때 사용되는 것이 BGP KeepAlive이다. 세션을 유지하고자 하는 시간을 hold time field에 초 단위로 담아보내게 되면 해당 시간만큼 세션의 수명이 갱신되는 식이다. 기본 hold time은 180초, 즉 3분이며 정상적인 BGP 세션에서는 hold time의 1/3이 지났을 때 다시 BGP KeepAlive 패킷을 보내 세션 연결을 연장하게 된다.

---

## 4. OSPF, Open Shortest Path First

OSPF는 AS 내부에서 패킷을 목적지까지 전달하는 방법, 즉 IGP(Interior Gateway Protocol)의 대표적인 방법이다. AS 간 라우팅 방식은 BGP가 유일하기 때문에 BGP 프로토콜 그 자체로 불리지만 IGP의 경우 종류가 다양하다. IGP는 다양한 방법론들이 적용되며 발전을 거듭해왔는데 최근에는 Link State Protocol 방식인 OSPF, IS-IS만이 사용된다고 할 수 있다.

### Dijkstra algorithm

BGP와 비교해 IGP의 가장 큰 차이점은 하나의 AS 내부에서 이뤄지는 라우팅이기 때문에 무조건 최단 거리를 찾고자 한다는 점이다. 수많은 라우터들이 연결되어있는 네트워크에서 최단거리를 찾는 문제는 알고리즘에서 그래프 문제라고 할 수 있다. OSPF는 그 중 가장 대표적인 그래프 알고리즘 중 하나인 Dijkstra algorithm을 사용한다. 따라서 모든 라우터와 이를 잇는 링크에 대한 정보들을 모두 알고 있다면 OSPF에서 최단거리를 찾는 연산의 시간 복잡도는 $$O(L+R \log R)$$이 된다. 여기서 $$L$$은 link의 개수, $$R$$은 라우터의 개수를 의미한다.

OSPF의 제1목적은 Dijkstra algorithm을 풀기 위한 정보를 AS 내부 라우터 간에 공유하는 것이다. OSPF 프로토콜 덕분에 모든 라우터들이 전체 AS의 네트워크 상태를 table 형태로 가질 수 있게 되며 최단 거리를 구할 수 있게 된다. Dijkstra algorithm을 풀기 위해서는 link의 cost를 정하는 것이 중요한데 일반적으로 cost는 속도를 기준으로 하며 100Mb를 1로 하여 구해진다. 1Gbps는 0.1, 10Mbps면 10인 식이다.

### Hierarchical Network

<img src="{{site.image_url}}/study/ospf_backbone.png" style="width:35em; display: block; margin: 0px auto;">

Dijkstra algorithm의 시간 복잡도가 상대적으로 낮다고 하더라도 라우터와 링크의 숫자가 크게 늘어난다면 느릴 수 밖에 없다. 이러한 문제를 해결하기 위해 AS 내부에서도 계층 구조를 형성하고, 특정 영역 내에서만 최단 거리를 찾는 연산을 수행하게 된다.

OSPF는 기본적으로 IP multicast를 통해 직접 연결된 라우터에 대해서만 전파된다. 따라서 다른 Area로 OSPF 패킷을 보내기 위해서는 다른 라우터가 포워딩해주어야 한다.

### LSA & LSDB

**LSA**(Link State Advertisement)란 OSPF에서 링크의 상태, 경로, 비용 등 최단 거리를 찾기 위해 필요한 여러 정보들을 담고 있는 패킷 형태의 데이터를 말한다. LSA는 그 자체로 전달될 수 없고, DBD, LSU등에 실려서 전달된다. 기본적으로 LSA는 수명이 1시간이며, 30분에 한 번씩 새롭게 전파된다.

LSA는 11개의 타입을 가지는데 대표적으로 다음과 같은 것들이 있다.

- Router LSA: AS 내의 모든 라우터가 전파하는 것으로, 자신과 연결된 링크들의 정보를 알리기 위해 사용된다.
- Network LSA: Router LSA로 모든 라우터가 전파하게 되면 과도하게 많은 패킷이 전달되는데, 이러한 비효율성을 막기 위해 대표 라우터(designated router)만이 LSA를 받아 정리하여 다른 Router에게 알리도록 할 수도 있다. 이때 사용되는 LSA이다.
- Summary LSA: 각 Area의 요약 정보를 다른 Area에 알리기 위해 사용된다. Area 간 경계에 있는 라우터만 전송하게 된다.

**LSDB**(Link State DataBase)는 이름 그대로 Link State를 저장하고 있는 table, database이다. 모든 라우터는 area 내의 LSA를 table 형태로 저장하고 이를 통해 최단 거리를 찾게 되는데 이를 LSDB라고 한다.

### OSPF Packet Type

OSPF 패킷은 총 5개의 타입을 가지고 있다.

|Type|Number|description|
|:------:|:---:|:---|
|**Hello**|1|라우터 간 연결을 유지하기 위해 사용|
|**DBD**|2|라우터가 가지고 있는 LSDB에 대한 요약 정보를 다른 라우터에 전달하기 위해 사용|
|**LSR**|3|수신한 DBD가 자신이 가진 것과 다른 경우 정확한 정보를 요청하기 위해 사용|
|**LSU**|4|LSR에 대한 응답으로 정확한 LSA를 알리기 위해 사용|
|**LSAck**|5|LSU를 정확하게 받았음을 알리기 위해 사용|

#### OSPF DBD

DBD(DataBase Description)은 라우터가 가지고 있는 LSDB에 대한 요약 정보를 다른 라우터에게 전달하기 위해 사용하는 OSPF 패킷이다. 여기서 말하는 요약 정보란 LSDB에 저장된 LSA들의 header를 말한다. 가지고 있는 모든 LSA를 주고 받기에는 크기가 너무 크기 때문에 header만을 전파하는 것이다. 다른 라우터로부터 DBD를 수신한 라우터는 자신이 가지고 있는 LSDB의 값들과 DBD의 값들을 비교하며 차이가 있는지 확인하는 작업을 수행하게 된다. 이때 만약 차이가 있다면 정확한 정보로 업데이트하기 위해 LSA를 처음 발신한 라우터에게 요청하게 되는데 이것이 LSR이다.

#### OSPF LSR, LSU, LSAck

LSR(Link State Request)는 위에서 말한 것과 같이 정확한 LSA를 얻기 위해 이를 요청하는 패킷이다. 라우터가 LSR을 받게 되면 자신이 가지고 있는 LSA를 담아 응답하게 되는데 이것이 LSU(Link State Update)이다. LSU를 응답으로 받은 라우터는 자신의 LSDB를 수신한 데이터에 맞춰 변경하고 주변의 라우터들에게 이 사실을 알린다. LSAck는 전파받은 다른 라우터들이 수신했음을 알리기 위해 사용된다.
