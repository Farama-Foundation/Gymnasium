---
layout: "contents"
title: Gym v0.23.1
---

# v0.23.1 Release Notes

This release contains a few small bug fixes and no breaking changes.

* Make ``VideoRecorder`` backward-compatible to ``gym<0.23`` by [@vwxyzjn](https://github.com/@vwxyzjn) in [#2678](https://github.com/openai/gym/pull/2678)
* Fix issues with pygame event handling (which should fix support on windows and in jupyter notebooks) by [@andrewtanJS](https://github.com/@andrewtanJS) in [#2684](https://github.com/openai/gym/pull/2684)
* Add py.typed to package_data by [@micimize](https://github.com/@micimize) in [#2683](https://github.com/openai/gym/pull/2683)
* Fixes around 1500 warnings in CI [@pseudo-rnd-thoughts](https://github.com/@pseudo-rnd-thoughts)
* Deprecation warnings correctly display now [@vwxyzjn](https://github.com/@vwxyzjn)
* Fix removing striker and thrower [@RushivArora](https://github.com/@RushivArora)
* Fix small dependency warning error [@ZhiqingXiao](https://github.com/@ZhiqingXiao)

**Full Changelog**: https://github.com/openai/gym/compare/0.23.0...0.23.1

**Github Release**: https://github.com/openai/gym/releases/tag/0.23.1
