---
layout: post
title: Customizing Vim
category_num : 7
keyword: '[Vim]'
---

# Customizing Vim

- update date: 2020.11.11
- environment: MacOS 10.15.6

## Install Vim

```bash
brew install vim
```

## Configuration for Vim

```bash
# .vimrc
syntax on

set encoding=utf-8
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
set backspace=indent,eol,start

highlight ColorColumn ctermbg=0 guibg=lightgrey
```

## Vim Plugins

```bash
# Install vim plugins
curl -fLo ~/.vim/autoload/plug.vim --create-dirs \
    https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim
```

```bash
# .vimrc
call plug#begin('~/.vim/plugged')
Plug 'morhetz/gruvbox' # color Theme
Plug 'vim-scripts/indentpython.vim' # indentation for python
Plug 'airblade/vim-gitgutter' # show git diff
call plug#end()

colorscheme gruvbox
set background=dark
```

```bash
# Vim Command mode on .vimrc
:PlugInstall
```

## Extra: Vim Commands

- `w` : move word
- `b` : move back word
- `gg` : first line of file
- `H` : first line of screen
- `:%` : last line of file
- `L` : last line of screen
- `:tabnew <file name>` : create new tab
- `:gt` : move next tab
- `:gT` : move previous tab
- `:#gt` : move to #th tab