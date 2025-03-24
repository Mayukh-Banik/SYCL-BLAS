
<h1 align="center">
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="https://www.hpc.cineca.it/wp-content/uploads/2023/10/SYCL.png" alt="Markdownify" width="200"></a>
  <br>
  SYCL-BLAS
  <br>
</h1>

<h4 align="center">An unofficial implementation of BLAS using SYCL</h4>

<!-- <p align="center">
  <a href="https://badge.fury.io/js/electron-markdownify">
    <img src="https://badge.fury.io/js/electron-markdownify.svg"
         alt="Gitter">
  </a>
  <a href="https://gitter.im/amitmerchant1990/electron-markdownify"><img src="https://badges.gitter.im/amitmerchant1990/electron-markdownify.svg"></a>
  <a href="https://saythanks.io/to/bullredeyes@gmail.com">
      <img src="https://img.shields.io/badge/SayThanks.io-%E2%98%BC-1EAEDB.svg">
  </a>
  <a href="https://www.paypal.me/AmitMerchant">
    <img src="https://img.shields.io/badge/$-donate-ff69b4.svg?maxAge=2592000&amp;style=flat">
  </a>
</p> -->

<p align="center">
  <a href="#key-features">Key Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#todo">TODO</a> •
  <!-- <a href="#download">Download</a> • -->
  <a href="#contributions">Contributions</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#license">License</a>
</p>



## Key Features

NetLib Conformant BLAS implementation with Tuning/Function selection determined at run/compile time for parallelized computation.

## Installation

This requires CMake Version > 3.28 (untested for earlier versions, change at your own risk). Build tool of your own choosing (Ninja, Make, etc). A sycl compatible compiler (ACPP, Intel, etc). A SYCL compatible device (everything except AMD consumer GPUS and CPUS).

```bash
# Meant to be used as a git submodule
git clone https://github.com/Mayukh-Banik/SYCL-BLAS
```

> **Note**
> AMD is broken on their CPUs for SYCL support, you must use AdaptiveCPP, there is no other option.

## TODO
AXPY functions are complete for basic case, everything else must be developed.

## Contributions

Current pull and push request formats are undecided, but any help is greatly appreciated! The end goal is to be like [CLBlast](https://github.com/CNugteren/CLBlast) where there's automatic tuning for SYCL platforms. This is very much a work in progress, and I'm planning on making this challenge Intel's MKL library, so any help is greatly appreciated, from documentation to multiplatform building.

## Credits

This software uses the following open source packages:

-[AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp)
<!-- - [Electron](http://electron.atom.io/)
- [Node.js](https://nodejs.org/)
- [Marked - a markdown parser](https://github.com/chjj/marked)
- [showdown](http://showdownjs.github.io/showdown/)
- [CodeMirror](http://codemirror.net/)
- Emojis are taken from [here](https://github.com/arvida/emoji-cheat-sheet.com)
- [highlight.js](https://highlightjs.org/) -->

<!-- ## Related

[Try Web version of Markdownify](https://notepad.js.org/markdown-editor/) -->

<!-- ## Support

<a href="https://buymeacoffee.com/amitmerchant" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/purple_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>

<p>Or</p> 

<a href="https://www.patreon.com/amitmerchant">
	<img src="https://c5.patreon.com/external/logo/become_a_patron_button@2x.png" width="160">
</a> -->

<!-- ## You may also like... -->
<!-- 
- [Pomolectron](https://github.com/amitmerchant1990/pomolectron) - A pomodoro app
- [Correo](https://github.com/amitmerchant1990/correo) - A menubar/taskbar Gmail App for Windows and macOS -->

## License

[Apache V2.0](https://github.com/Mayukh-Banik/SYCL-BLAS/blob/main/LICENSE)

---

> GitHub [@Mayukh-Banik](https://github.com/Mayukh-Banik) &nbsp;&middot;&nbsp;
> Email mayukh1banik@gmail.com (Contact for Job Offers/Miscellaneous queries)

