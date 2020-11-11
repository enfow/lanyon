---
layout: post
title: Customizing Vim
category_num : 7:
keyword: '[Vim]'
---

# Customizing Vim

- update date: 2020.11.11
- environment: MacOS 10.15.6

```bash
brew install vim
```

```bash
syntax on
set noerrorbells
set tabstop=4
set softtabstop=4
set expandtab
set smartindent
set nu
set nowrap
set smartcase
set noswapfile
set nobackup
set incsearch
set colorcolumn=80
highlight ColorColumn ctermbg=0 guibg=lightgrey
```

## Vim Plugins

```bash
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

:PlugInstall

```bash
call plug#begin('~/.vim/plugged')
Plug 'morhetz/gruvbox'
Plug 'vim-scripts/indentpython.vim'
call plug#end()

colorscheme gruvbox
set background=dark
```
