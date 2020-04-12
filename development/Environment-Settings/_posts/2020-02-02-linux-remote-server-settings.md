---
layout: post
title: Linux remote server settings
---

# 리눅스 원격 서버 셋팅하기

- update date : 20.02.03

## 0. ENVIRONMENT

- server : ubuntu linux 18.04
- client : macOS Mojave

## 1. SSH SERVER INSTALLATION

SSH는 Secure shell의 약자로, 네트워크에 연결된 원격 시스템에 접속할 수 있도록 해준다. Talnet 등 비슷한 기능을 제공하는 프로토콜이 있지만, 보안 상의 문제로 SSH를 많이 사용한다.

### openssh-server install

- Ubuntu linux에는 SSH client가 기본 설치되어 있으나 server는 별도로 설치가 필요하다.

```
sudo apt-get install openssh-server
```

## 2. SSH CONFIGURATION SETTINGS

### 1) SERVER: sshd_config file

SSH 셋팅은 `/etc/ssh` 디렉토리에 저장된 `sshd_config`에서 할 수 있다. 해당 파일에는 주석처리되어 있는 다양한 configuration이 저장되어 있는데, 필요한 내용만 주석 해제하여 설정하면 된다.

설정을 바꾸고나면 아래 명령어를 통해 ssh를 다시 시작해주어야 한다.

```
$
sudo service ssh restart
```

#### - Port

SSH의 기본 포트 번호는 22이다. 보안 상의 이유로 변경하는 경우가 많다.

```
# /etc/ssh/sshd_config

#Port 22
```

복수의 포트 번호를 입력하기 위해서는 다음과 같이 여러 개를 사용한다.

```
# /etc/ssh/sshd_config

Port 43434
Port 43435
Port 43436
```

jupyer notebook 등 시스템의 다른 프로그램이 사용하는 포트로 설정될 경우 문제가 발생할 수 있으므로 5자리 숫자로 설정하는 것이 좋다.

#### - PermitRootLogin

root로의 로그인 허용 여부를 설정하는 configuration이다. 기본적으로는 보안 상의 이유로 root 로그인을 허용하지 않는다.

```
# /etc/ssh/sshd_config

#PermitRootLogin prohibit-password
```

이를

```
# /etc/ssh/sshd_config

PermitRootLogin yes
```

로 바꾸면 root 로그인이 가능해진다.

#### - PasswordAuthentication

Password를 통한 인증 여부를 설정할 수 있다.

```
# /etc/ssh/sshd_config

PasswordAuthentication no
```

위와 같이 변경할 경우 password로 접속이 불가능해진다.

#### - PubkeyAuthentication

Public Key를 통한 인증 여부를 설정할 수 있다.

```
# /etc/ssh/sshd_config

PubkeyAuthentication yes
```

와 같이 설정하면 Public Key를 통한 인증이 가능해진다.

### 2) Public Key

SSH는 기본적으로 id-password를 통해 접속할 수 있다. 하지만 password의 입력 없이 자동적으로 접속도 가능한데, 이때 사용하는 것이 public key이다. 위에서 언급한 것과 같이 Public Key를 사용하기 위해서는 sshd configuration를 설정해주어야 한다.

Public Key는 client가 보내는 Public Key를 server가 허용하고 있는 Public Key와 대조하는 방식으로 동작한다. 따라서 Client에서 우선 Key를 생성하고 이를 Server에 저장하는 순서로 이뤄진다.

#### (1) Client - Create Public Key

root 계정을 사용하여 `~./ssh/` 디렉토리에서 Key를 생성한다.

```
ssh-keygen -t rsa
```

위의 명령어를 실행하면 `id_rsa`, `id_rsa.pub` 두 개의 파일이 생성된다. 이 중 Public Key는 `id_rsa.pub`이며, 여기에 담긴 내용을 서버에 저장해야 한다. 

참고로 `$ ssh-kegen -t dsa` 를 통해 dsa 암호를 만들 수도 있다. 하지만 dsa는 rsa보다 보안성이 떨어지므로 사용하지 않는 것이 좋고, openssh 버전에 따라서는 따로 설정을 바꾸어주어야 한다([link](<https://unix.stackexchange.com/questions/247612/ssh-keeps-skipping-my-pubkey-and-asking-for-a-password>)).

#### (2) SERVER - Save Public Key as authorized_keys

클라이언트의 `id_rsa.pub` 파일을 사용자 디렉토리 내에 저장해야 한다. root의 `/etc/ssh/` 디렉토리에 저장한다고 해서 사용자별 접속 시에 적용되지 않는다. 개별 사용자를 단위로 해야 하며, 이를 위해 `.ssh\` 디렉토리를 사용자 디렉토리에 생성하고 내부에 authorized_keys 파일을 생성한 후 복사 붙여넣기 해 준다.

이 과정은 client에서 아래 명령어를 통해서도 가능하다. 사용자 디렉토리에 `.ssh\` 디렉토리의 생성부터 `authroized_keys`로 파일 명을 변환하여 저장하는 것까지 된다.

```
ssh-copy-id <host_name>@<host_id_address> -p <port_number>
```

단 SERVER에서 password 로 ssh 접속을 막아둔 경우라면 `ssh-copy-id` 로 불가능하다. `etc/ssh/sshd-config` 에서 설정을 바꾸어주어야 한다.

### 3) CLIENT: config file

Client의 `~/.ssh/` 디렉토리에 저장되어 있는 `config` 파일에서 호스트 정보를 설정할 수 있다. 이를 통해 `ssh <host_name>` 으로 간편하게 서버에 접속할 수 있다.

```
# ~/.ssh/config

Host <host_name>
    Host <host_ip_address/url>
    User <host_user>
    Port <port_number>
    IdentityFile ~/.ssh/id_rsa
```

IdentitiyFile은 `id_rsa.pub`이 아니라 `id_rsa` 파일로 설정해준다.

그리고 아래 두 옵션을 추가하면 ssh 연결 지속 시간을 늘릴 수 있다.

```
    ServerAliveInterval 15
    ServerAliveCountMax 3
```

### 4) SERVER: 방화벽 설정

보다 안전한 ssh 사용을 위해서는 방화벽 설정을 통해 특정 포트로만 접근이 가능하도록 설정할 필요가 있다. 이는 Ubuntu Linux의 기본 방화벽인 **UFW**를 통해 가능하다.

**UFW 방화벽 활성화**

```
sudo ufw enable
```

**특정 포트 허용**

```
sudo ufw allow <port_num>
```

**특정 tcp 포트만 허용**

```
sudo ufw allow <port_num>/tcp
```

- ssh를 위해서는 tcp 포트만으로도 가능하므로 보안상 tcp 포트만 열어주는 것이 좋다.

**특정 포트 거부**

```
sudo ufw deny <port_num>
```

**특정 룰의 삭제**

```
sudo ufw delete allow 22
```

- 22번 포트를 허용하는 룰이 있을 때 이를 삭제하는 명령어이다.

## 3. ACCESS

  **ip 주소로 접속하기**

  ```
  ssh <host_user>@<host_ip_address> -p <port_num>
  ```

  **config file 설정값 이용하기**

  ```
  ssh <host_name>
  ```
