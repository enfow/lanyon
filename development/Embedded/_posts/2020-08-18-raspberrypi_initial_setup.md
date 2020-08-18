---
layout: post
title: Raspberry Pi initial Setup
subtitle: Install Ubuntu & Connect WiFi
keyword: '[RaspberryPi]'
category_num : 1
---

# Raspberry Pi initial Setup

- Raspberry 4
- update at 2020.08.18

## 1. Install Ubuntu

라즈베리파이는 마이크로 컴퓨터로 불리며 리눅스 같은 컴퓨터 운영체제를 설치할 수 있다. 우분투 리눅스를 설치하기 위해서는 운영체제 이미지를 가져올 Micro SD 카드가 필요하며 구체적인 설치 방법은 다음과 같다.

### 1.1. Download Ubuntu Linux for Raspberry Pi

우선 라즈베리파이에 설치할 수 있는 우분투 라즈베리 파이 이미지를 다운로드 받아야 한다. [홈페이지](<(<https://ubuntu.com/download/raspberry-pi>)에 들어가서 아래 항목 중 자신의 라즈베리 파이 버전에 맞는 이미지를 다운로드 받으면 된다.

<img src="{{site.image_url}}/development/raspberry_pi_ubuntu_page.png" style="width:35em; display: block; margin: 0em auto">

### 1.2. Writing Image File

[Balena Etcher](<https://www.balena.io/etcher/>)는 `.iso`, `.img`와 같은 이미지 파일을 이동식 저장장치에 쓰는 프로그램이다. 이를 이용하면 쉽게 우분투 리눅스 설치 Micro SD 카드를 만들 수 있다.

<img src="{{site.image_url}}/development/balena_etcher.png" style="width:35em; display: block; margin: 0em auto">

프로그램을 다운로드받고 실행하면 다음과 같은 화면이 뜨는데 설치하고자 하는 이미지 파일과 이미지가 저장될 Micro SD 카드를 차례대로 지정해주기만 하면 된다.

### 1.3 Install Ubuntu

Micro SD 카드를 라즈베리파이에 장착한 뒤 전원을 연결해주기만 하면 된다. 초기 Username과 Password는 모두 `ubuntu`로 설정되어 있다.

---

## 2. Internet Setup

Ethernet은 공유기에 LAN선으로 라즈베리파이와 연결해주면 자동으로 setup이 되므로 쉽게 사용할 수 있다. 여기서는 커맨드라인만을 사용하여 WiFi를 연결하는 방법에 대해 다루고자 한다.

### 2.1. Check WiFi Configuration

Ethernet의 연결 여부, IP 주소 등을 확인할 때에는 `ifconfig` 명령어를 사용한다. WiFi와 같은 무선 네트워크의 경우 `iwconfig`를 사용하는데 처음 리눅스를 설치한 경우 기본으로 깔려 있지 않으므로 다음 명령어를 사용하여 설치해주어야 한다.

- `sudo apt-get install wireless-tools`

설치 이후 `iwconfig`를 입력하면 무선 네트워크 인터페이스를 다수 확인할 수 있다. 이 중 WiFi는 `wlan0` 인터페이스를 사용한다. WiFi 연결이 되어 있지 않으므로 

```
Access Point: Not-Associated 
```

를 비롯하여 대부분의 기본 정보들을 확인할 수 없다. WiFi가 정상적으로 연결되면 위의 Access Point에 IP 주소가 채워질 것이다.

### 2.2. Search WiFi Network

WiFi를 사용하기 위해서는 `wlan0` 인터페이스에 연결할 WiFi 네트워크를 찾아야 한다. 이를 위해 `wlan0` 인터페이스를 활성화 해준다. 아래에 나와있듯 `iwconfig`가 아니라 `ifconfig`를 사용해야 한다.

- `sudo ifconfig wlan0 up`

연결 가능한 WiFi 네트워크는 다음 명령어로 확인 가능하다.

- `sudo iwlist wlan0 scan`

이를 통해 확인하고자 하는 정보는 두 가지로, 인터페이스의 이름은 무엇인지와 보안 정책은 무엇인지이다. 하지만 위의 명령어를 곧바로 입력했을 때 경우에 따라서는 매우 많은 네트워크가 출력되어 원하는 정보를 확인하기 어려울 수 있다. 따라서 다음과 같이 필요한 정보만을 차례대로 출력할 수 있다.

- `sudo iwlist wlan0 scan | grep ESSID`
- `sudo iwlist wlan0 scan | grep IEEE 802.11`

첫 번째 `ESSID`로는 네트워크의 이름을 확인한다. 노트북으로 연결할 때 사용하는 WiFi 이름과 동일하다. 두 번째의 경우 `WPA2`를 사용하면 다음과 같이 나온다.

```
IE: IEEE 802.11i/WPA2 Version 1
```

여러 개가 나오는 경우 사용하고자 하는 네트워크의 정보만 알고 있으면 된다.

### 2.3. Settigs for WPA2

WPA의 경우 암호를 요구하는데 미리 암호를 저장해두고 사용하는 것이 가능하다. 다만 암호를 그대로 저장하여 관리하는 것은 다소 위험하므로 암호화하여 저장하는 것이 일반적이다. 참고로 암호는 아래 파일로 저장/관리하게 된다. 

```
/etc/wpa_supplicant/wpa_supplicant.conf
```

암호를 만들기 위해서는 네트워크의 아이디와 비밀번호를 알아야 한다. WiFi를 사용할 때 아이디와 비밀번호를 입력하는 것과 동일하다. 다음 명령어를 입력하면 위의 파일이 생성된다.

- `wpa_passphrase <WiFi ID> <WiFi Password> > /etc/wpa_supplicant/wpa_supplicant.conf`

그 내용을 확인해보면 다음과 같다.

```
network={
    ssid="<WiFi ID>"
    #psk="<WiFi Password>"
    psk=<KEY>
}
```

여기서 `#psk`는 더 이상 필요하지 않으므로 삭제해준다.

### 2.4 Connect WiFi Network

연결을 위해서는 다음 명령어를 차례대로 입력한다.

- `iwconfig wlan0 essid <WiFi ID>`
- `wpa_supplicant -iwlan0 -c /etc/wpa_supplicant/wpa_supplicant.conf &`
- `dhcpcd wlan0 &`

dhcp 동적 할당까지 마무리하면 `iwconfig` 명령어로 확인했을 때 처음과는 달리 Access Point에 IP 주소가 채워져 있는 것을 확인할 수 있을 것이다.

- `ping www.google.com`

명령어를 사용하여 구글에 접속할 수 있는지 확인할 수 있다. 만약 

```
Temporary failure in name resolution
```

와 같은 에러가 발생한다면 Nameserver 문제이므로 /etc/resolv.conf 파일에 다음 내용을 추가해준다.

```
nameserver 8.8.8.8
```
