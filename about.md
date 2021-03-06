---
layout: page
title: About
permalink: /about/
---

# Humanitexts
## Use Technology for Higher Things

- Present dependencies for _HumaniTexts_
  - [zettelgeist.com](http://zettelgeist.com/) for all research (you will see)
  - [vim.org](https://www.vim.org/) with very few plugins
    - [VOoM](https://www.vim.org/scripts/script.php?script_id=2657) for outlining in Vim (as powerful as Microsoft Word outlining) 
    - [marvim.vim](https://github.com/vim-scripts/marvim/blob/master/plugin/marvim.vim) for making macros in Vim 
    - [markdown-preview.vim](https://github.com/iamcco/markdown-preview.vim) 
    - [NERDTree](https://www.vim.org/scripts/script.php?script_id=1658) for file managing in Vim 
  - [vifm](https://vifm.info/) for file managing in place 
  - [gawk](https://www.gnu.org/software/gawk/) for some search and replace operations  
  - [zotero](https://www.zotero.org/) for bibliography management 
  - [better-bibtex](https://github.com/retorquere/zotero-better-bibtex) for integrating zotero with bibtex 
  - [pandoc](https://pandoc.org/) for conversion of your markdown to whatever "they" need 
  - [github](github.com) for ... everything that git does.

- Useful command line tools to deploy in order to stay focused  
  - [mutt](http://www.mutt.org/) for email 
  - [slack-cli](https://pypi.python.org/pypi/slack-cli/2.0.3) for collaborating 
  - [lynx](https://lynx.browser.org/) for searching 
  - [hangups](https://github.com/tdryer/hangups) for chatting with outside world (with google voice and hangouts until android messenger works) 
  - [ghi](https://github.com/stephencelis/ghi) for managing github issues
  - tilix, iterm2, or tmux to return to panes and session windows (you don't need Windows to have windows - they are built into the OS - always were)  

- For the rest, in vim style
  - [vimium](https://vimium.github.io/) for vim keyboard shortcuts in chrome 
  - [wasavi](http://appsweets.net/wasavi/) to use vim editor to write email, fill out forms, etc. 
  
 - Tools that this method replaces 
   - _Microsoft Word,_ including Outlining (Great wysiwyg writing environment, but no notetaking and search capabilities.  Outlining feature probably the best, but it can be replicated in Vim with VOoM.)
   - _Procite, RefWorks, Endnote, Mendeley_ (Bibliographical management softwares, sometimes mistaken to be notetaking programs. Some manipulation of "notes" features can make them usable for research, but really they are designed for inputting bibliography and linking to foot- and end-notes in Word and GDocs. The same can be said of Zotero: though it is still an excellent bibliography manager, its "notes" system is not designed for research notetaking. See comment on _Scribe_ below.)
   - _Onenote, Evernote_ (Fine data gathering platforms, but not good data management systems. No real search and output capabilities.)
   - [Scrivener](https://www.literatureandlatte.com/scrivener/overview) (A favorite for many writers and researchers.  But the notetaking function, despite the corkboard system, is not a full data management system.)
   - [Zettelkasten3](http://zettelkasten.danielluedecke.de/en/) (An excellent program that does integrate notetaking, bibliography, outlining, and writing, but is limited because it slows down as more material is added.) 
   - [NoodleTools](https://www.noodletools.com/) (An excellent online iteration of a notetaking, outlining, and writing system.  Designed for targeted users of K-12.  Recommended for teachers and parents who want their students to learn the fundamental steps of research. Very nice interface with virtual notecards, sorting table, etc. Just not aimed at college-level or professional users.)
   - [SuperNotecard](https://www.supernotecard.com/app/index.php?project=list) (Nice interface replicating real notecards and sorting desk. Limited for data analysis.)
   - [The Outliner of Giants](https://www.theoutlinerofgiants.com/) (Outlining connected to Google Docs, but every node separate. VOoM in Vim surpasses it.)
   - [Scribe](https://forums.zotero.org/discussion/829/zotero-and-scribe-for-historians-especially) (No longer available, but it was the project that came closest to providing a full platform for notetaking, bibliographical management, outlining, and writing for serious scholarship.  Sponsored by Roy Rosenzweig Center for History and New Media. Developer Elena Razlogova passed the program on to Zotero. Many continue to hope that Zotero will gain better notetaking features that existed in Scribe.)
   - _Squarenote_ (No longer available, but it was a wonderful DOS program that never made the transition to Windows. [Review from 1991.](https://drive.google.com/file/d/1AKvFXsa_sniTBIjyQAEcsSkl7EIGZMiu/view?usp=sharing) [Review from 1993.](https://drive.google.com/file/d/1ckViIW9ucqarra1sTNCD_n2cSIMRHRT5/view?usp=sharing))
   - _Atom, Sublime Text 3_ and other IDEs.  See comment above about tilix, iterm2, tmux, etc. 


# The Humanitexts Suite Setup

Based on a photo of Tilix in action:  

The top tiles are for processing the books.  Ignore.  

The bottom three are, from left to right:

L - the hoc-sandbox directory opened in Vim with NERDTree on left.  This lets you go to specific "pages" and look at them in context.  Let's say zfind has some interesting material on Rheingold-0325.  Use NERDTree to get into Rheingold folder, then find 0325.  Then you can read 0326, 0327..., and either add tags there or yank and put into new zettels elsewhere.  The key point, here, is that NERDTree does let you just open to "preview" as you wish.... This fills out the zfind and zfilter functions, when you want to see the relevant snippets in context.

C - the center screen is a zfind result, then "sorted" with the sort script I made, and reading it in "less."  This is where you see what came out of a zfind search.  and you can then look to the Left tile if you want to see what comes on the next pages.  

R - That is the Book Outline with :Voom markdown open.  You can write notes about what you are finding in the other two tiles, or even  yank and put things right in there. (Though we will probably want to make new zettels or add tags for future searches.)

