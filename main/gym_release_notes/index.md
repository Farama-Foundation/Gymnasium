# Gym Release Notes

## 0.26.2

*Released on 2022-10-04 - [GitHub](https://github.com/openai/gym/releases/tag/0.26.2) - [PyPI](https://pypi.org/project/gym/0.26.2/)*

<h2>Release notes</h2>
<p>This is another very minor bug release.</p>
<h3>Bugs Fixes</h3>
<ul>
<li>As <code>reset</code> now returns <code>(obs, info)</code> then in the vector environments, this caused the final <code>step</code>'s info to be overwritten. Now, the final observation and info are contained within the info as "final_observation" and "final_info" <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Adds warnings when trying to render without specifying the <code>render_mode</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
<li>Updates Atari Preprocessing such that the wrapper can be pickled <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vermouth1992/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vermouth1992">@vermouth1992</a></li>
<li>Github CI was hardened to such that the CI just has read permissions <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/sashashura/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/sashashura">@sashashura</a></li>
<li>Clarify and fix typo in <code>GraphInstance</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/ekalosak/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/ekalosak">@ekalosak</a></li>
</ul>

## 0.26.1

*Released on 2022-09-16 - [GitHub](https://github.com/openai/gym/releases/tag/0.26.1) - [PyPI](https://pypi.org/project/gym/0.26.1/)*

<h2>Release Notes</h2>
<p>This is a very minor bug fix release for 0.26.0</p>
<h3>Bug Fixes</h3>
<ul>
<li><a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1364336791" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3072" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3072/hovercard" href="https://github.com/openai/gym/pull/3072">#3072</a> - Previously <code>mujoco</code> was a necessary module even if only <code>mujoco-py</code> was used. This has been fixed to allow only <code>mujoco-py</code> to be installed and used. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/YouJiacheng/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/YouJiacheng">@YouJiacheng</a></li>
<li><a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1366244408" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3076" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3076/hovercard" href="https://github.com/openai/gym/pull/3076">#3076</a> - <code>PixelObservationWrapper</code> raises an exception if the <code>env.render_mode</code> is not specified. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vmoens/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vmoens">@vmoens</a></li>
<li><a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1366720706" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3080" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3080/hovercard" href="https://github.com/openai/gym/pull/3080">#3080</a> - Fixed bug in <code>CarRacing</code> where the colour of the wheels were not correct <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/foxik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/foxik">@foxik</a></li>
<li><a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1368898455" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3083" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3083/hovercard" href="https://github.com/openai/gym/pull/3083">#3083</a> - Fixed BipedalWalker where if the agent moved backwards then the rendered arrays would be a different size. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
</ul>
<h3>Spelling</h3>
<ul>
<li>Fixed truncation typo in readme API example <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/rdnfn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/rdnfn">@rdnfn</a></li>
<li>Updated pendulum observation space from angle to theta to make more consistent <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/ikamensh/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/ikamensh">@ikamensh</a></li>
</ul>

## 0.26.0

*Released on 2022-09-06 - [GitHub](https://github.com/openai/gym/releases/tag/0.26.0) - [PyPI](https://pypi.org/project/gym/0.26.0/)*

<h1>Release notes for v0.26.0</h1>
<p>This release is aimed to be the last of the major API changes to the core API. All of the previously "turned off" changes of the base API (step termination / truncation, reset info, no seed function, render mode determined by initialization) are now expected by default. We still plan to make breaking changes to Gym itself, but to things that are very easy to upgrade (environments and wrappers), and things that aren't super commonly used (the vector API). Once those aspects are stabilized, we'll do a proper 1.0 release and follow semantic versioning. Additionally, unless something goes terribly wrong with this release and we have to release a patched version, this will be the last release of Gym for a while.</p>
<p>If you've been waiting for a "stable" release of Gym to upgrade your project given all the changes that have been going on, this is the one.</p>
<p>We also just wanted to say that we tremendously appreciate the communities patience with us as we've gone on this journey taking over the maintenance of Gym and making all of these huge changes to the core API. We appreciate your patience and support, but hopefully, all the changes from here on out will be much more minor.</p>
<h2>Breaking backward compatibility</h2>
<p>These changes are true of all <strong>gym</strong>'s internal wrappers and environments but for environments not updated, we provide the <code>EnvCompatibility </code> wrapper for users to convert old gym v21 / 22 environments to the new core API. This wrapper can be easily applied in <code>gym.make</code> and <code>gym.register</code> through the <code>apply_api_compatibility </code> parameters.</p>
<ul>
<li><code>Step</code> Termination / truncation - The <code>Env.step</code> function returns 5 values instead of 4 previously (<code>observations, reward, termination, truncation, info</code>). A blog with more details will be released soon to explain this decision. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/arjun-kg/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/arjun-kg">@arjun-kg</a></li>
<li>Reset info - The <code>Env.reset</code> function returns two values (<code>obs</code> and <code>info</code>) with no <code>return_info</code> parameter for gym wrappers and environments. This is important for some environments that provided action masking information for each actions which was not possible for resets. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/balisujohn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/balisujohn">@balisujohn</a></li>
<li>No <code>Seed</code> function - While <code>Env.seed</code> was a helpful function, this was almost solely used for the beginning of the episode and is added to <code>gym.reset(seed=...)</code>. In addition, for several environments like Atari that utilise external random number generators, it was not possible to set the seed at any time other than <code>reset</code>. Therefore, <code>seed</code> is no longer expected to function within gym environments and is removed from all gym environments <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/balisujohn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/balisujohn">@balisujohn</a></li>
<li>Rendering - It is normal to only use a single render mode and to help open and close the rendering window, we have changed <code>Env.render</code> to not take any arguments and so all render arguments can be part of the environment's constructor i.e., <code>gym.make("CartPole-v1", render_mode="human")</code>. For more detail on the new API, see <a href="https://younis.dev/blog/render-api/" rel="nofollow">blog post</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
</ul>
<h2>Major changes</h2>
<ul>
<li>Render modes - In <code>v25</code>, there was a change in the meaning of render modes, i.e. "rgb_array" returned a list of rendered frames with "single_rgb_array" returned a single frame. This has been <strong>reverted</strong> in this release with "rgb_array" having the same meaning as previously to return a single frame with a new mode "rgb_array_list" returning a list of RGB arrays. The capability to return a list of rendering observations achieved through a wrapper applied during <code>gym.make</code>. <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1348216259" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3040" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3040/hovercard" href="https://github.com/openai/gym/pull/3040">#3040</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
<li>Added <code>save_video</code> that uses <code>moviepy</code> to render a list of RGB frames and updated <code>RecordVideo</code> to use this function. This removes support for recording <code>ansi</code> outputs.  <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1325626729" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3016" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3016/hovercard" href="https://github.com/openai/gym/pull/3016">#3016</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
<li><code>RandomNumberGenerator</code> functions: <code>rand</code>, <code>randn</code>, <code>randint</code>, <code>get_state</code>, <code>set_state</code>, <code>hash_seed</code>, <code>create_seed</code>, <code>_bigint_from_bytes</code> and <code>_int_list_from_bigint</code> have been removed. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/balisujohn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/balisujohn">@balisujohn</a></li>
<li>Bump <code>ale-py</code> to <code>0.8.0</code> which is compatibility with the new core API</li>
<li>Added <code>EnvAPICompatibility</code> wrapper <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
</ul>
<h2>Minor changes</h2>
<ul>
<li>Added improved <code>Sequence</code>, <code>Graph</code> and <code>Text</code> sample masking <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Improved the gym <code>make</code> and <code>register</code> type hinting with <code>entry_point</code> being a necessary parameter of <code>register</code>. <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1348658998" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3041" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3041/hovercard" href="https://github.com/openai/gym/pull/3041">#3041</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Changed all URL to the new gym website <a href="https://www.gymlibrary.dev/" rel="nofollow">https://www.gymlibrary.dev/</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/FieteO/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/FieteO">@FieteO</a></li>
<li>Fixed mujoco offscreen rendering with weight and height value &gt; 500 <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1349697695" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3044" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3044/hovercard" href="https://github.com/openai/gym/pull/3044">#3044</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/YouJiacheng/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/YouJiacheng">@YouJiacheng</a></li>
<li>Allowed toy_text environment to render on headless machines <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1346463189" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3037" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3037/hovercard" href="https://github.com/openai/gym/pull/3037">#3037</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
<li>Renamed the motors in the mujoco swimmer envs <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1345599172" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/3036" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/3036/hovercard" href="https://github.com/openai/gym/pull/3036">#3036</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/lin826/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/lin826">@lin826</a></li>
</ul>

## 0.25.2

*Released on 2022-08-18 - [GitHub](https://github.com/openai/gym/releases/tag/0.25.2) - [PyPI](https://pypi.org/project/gym/0.25.2/)*

<h1>Release notes for v0.25.2</h1>
<p>This is a fairly minor bug fix release.</p>
<h2>Bug Fixes</h2>
<ul>
<li>Removes requirements for <code>_TimeLimit.truncated</code> in info for step compatibility functions. This makes the step compatible with Envpool <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/arjun-kg/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/arjun-kg">@arjun-kg</a></li>
<li>As the ordering of <code>Dict</code> spaces matters when flattening spaces, updated the <code>_\_eq_\_</code> to account for the <code>.keys()</code> ordering. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/XuehaiPan/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/XuehaiPan">@XuehaiPan</a></li>
<li>Allows <code>CarRacing</code> environment to be pickled. Updated all gym environments to be correctly pickled. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
<li>seeding <code>Dict</code> and <code>Tuple</code> spaces with integers can cause lower-specification computers to hang due to requiring 8Gb memory. Updated the seeding with integers to not require unique subseeds (subseed collisions are rare). For users that require unique subseeds for all subspaces, we recommend using a dictionary or tuple with the subseeds. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/olipinski/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/olipinski">@olipinski</a></li>
<li>Fixed the metaclass implementation for the new render api to allow custom environments to use metaclasses as well. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/YouJiacheng/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/YouJiacheng">@YouJiacheng</a></li>
</ul>
<h2>Updates</h2>
<ul>
<li>Simplifies the step compatibility functions to make them easier to debug. Time limit wrapper with the old step API favours terminated over truncated if both are true. This is as the old done step API can only encode 3 states (cannot encode <code>terminated=True</code> and <code>truncated=True</code>) therefore we must encode to only <code>terminated=True</code> or <code>truncated=True</code>.  <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Add Swig as a dependency <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/kir0ul/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/kir0ul">@kir0ul</a></li>
<li>Add type annotation for <code>render_mode</code> and <code>metadata</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/bkrl/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/bkrl">@bkrl</a></li>
</ul>

## 0.25.1

*Released on 2022-07-26 - [GitHub](https://github.com/openai/gym/releases/tag/0.25.1) - [PyPI](https://pypi.org/project/gym/0.25.1/)*

<h1>Release notes</h1>
<ul>
<li>Added rendering for CliffWalking environment <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
<li><code>PixelObservationWrapper</code> only supports the new render API due to difficulty in supporting both old and new APIs. A warning is raised if the user is using the old API <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vmoens/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vmoens">@vmoens</a></li>
</ul>
<h3>Bug fix</h3>
<ul>
<li>Revert an incorrect edition on wrapper.FrameStack  <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/ZhiqingXiao/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/ZhiqingXiao">@ZhiqingXiao</a></li>
<li>Fix reset bounds for mountain car <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/psc-g/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/psc-g">@psc-g</a></li>
<li>Removed skipped tests causing bugs not to be caught <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Added backward compatibility for environments without metadata <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Fixed <code>BipedalWalker</code> rendering for RGB arrays <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/1b15/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/1b15">@1b15</a></li>
<li>Fixed bug in <code>PixelObsWrapper</code> for using the new rendering <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/younik/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/younik">@younik</a></li>
</ul>
<h3>Typos</h3>
<ul>
<li>Rephrase observations' definition in Lunar Lander Environment @EvanMath</li>
<li>Top-docstring in <code>gym/spaces/dict.py</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Ice1187/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Ice1187">@Ice1187</a></li>
<li>Several typos in <code>humanoidstandup_v4.py</code>, <code>mujoco_env.py</code>, and <code>vector_list_info.py</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/timgates42/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/timgates42">@timgates42</a></li>
<li>Typos in passive environment checker <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Typos in Swimmer rotations <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/lin826/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/lin826">@lin826</a></li>
</ul>

## 0.25.0

*Released on 2022-07-13 - [GitHub](https://github.com/openai/gym/releases/tag/0.25.0) - [PyPI](https://pypi.org/project/gym/0.25.0/)*

<h1>Release notes</h1>
<p>This release finally introduces all new API changes that have been planned for the past year or more, all of which will be turned on by default in a subsequent release. After this point, Gym development should get massively smoother. This release also fixes large bugs present in 0.24.0 and 0.24.1, and we highly discourage using those releases.</p>
<h2>API Changes</h2>
<ul>
<li><code>Step</code> - A majority of deep reinforcement learning algorithm implementations are incorrect due to an important difference in theory and practice as <code>done</code> is not equivalent to <code>termination</code>. As a result, we have modified the <code>step</code> function to return five values, <code>obs, reward, termination, truncation, info</code>. The full theoretical and practical reason (along with example code changes) for these changes will be explained in a soon-to-be-released blog post. The aim for the change to be backward compatible (for now), for issues, please put report the issue on github or the discord. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/arjun-kg/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/arjun-kg">@arjun-kg</a></li>
<li><code>Render</code> - The render API is changed such that the mode has to be specified during <code>gym.make</code> with the keyword <code>render_mode</code>, after which, the render mode is fixed. For further details see <a href="https://younis.dev/blog/2022/render-api/" rel="nofollow">https://younis.dev/blog/2022/render-api/</a> and <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1160674687" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/2671" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/2671/hovercard" href="https://github.com/openai/gym/pull/2671">#2671</a>. This has the additional changes
<ul>
<li>with <code>render_mode="human"</code> you don't need to call .render(), rendering will happen automatically on <code>env.step()</code></li>
<li>with <code>render_mode="rgb_array"</code>, <code>.render()</code> pops the list of frames rendered since the last <code>.reset()</code></li>
<li>with <code>render_mode="single_rgb_array"</code>, <code>.render()</code> returns a single frame, like before.</li>
</ul>
</li>
<li><code>Space.sample(mask=...)</code> allows a mask when sampling actions to enable/disable certain actions from being randomly sampled. We recommend developers add this to the <code>info</code> parameter returned by <code>reset(return_info=True)</code> and <code>step</code>. See <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1275024912" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/2906" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/2906/hovercard" href="https://github.com/openai/gym/pull/2906">#2906</a> for example implementations of the masks or the individual spaces. We have added an example version of this in the taxi environment. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Add <code>Graph</code> for environments that use graph style observation or action spaces. Currently, the node and edge spaces can only be <code>Box</code> or <code>Discrete</code> spaces. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jjshoots/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jjshoots">@jjshoots</a></li>
<li>Add <code>Text</code> space for Reinforcement Learning that involves communication between agents and have dynamic length messages (otherwise <code>MultiDiscrete</code> can be used). <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/ryanrudes/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/ryanrudes">@ryanrudes</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
</ul>
<h2>Bug fixes</h2>
<ul>
<li>Fixed car racing termination where if the agent finishes the final lap, then the environment ends through truncation not termination. This added a version bump to Car racing to v2 and removed Car racing discrete in favour of <code>gym.make("CarRacing-v2", continuous=False)</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/araffin/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/araffin">@araffin</a></li>
<li>In <code>v0.24.0</code>, <code>opencv-python</code> was an accidental requirement for the project. This has been reverted. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/KexianShen/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/KexianShen">@KexianShen</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Updated <code>utils.play</code> such that if the environment specifies <code>keys_to_action</code>, the function will automatically use that data. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Markus28/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Markus28">@Markus28</a></li>
<li>When rendering the blackjack environment, fixed bug where rendering would change the dealers top car. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/balisujohn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/balisujohn">@balisujohn</a></li>
<li>Updated mujoco docstring to reflect changes that were accidently overwritten. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Markus28/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Markus28">@Markus28</a></li>
</ul>
<h2>Misc</h2>
<ul>
<li>The whole project is partially type hinted using <a href="https://github.com/microsoft/pyright">pyright</a> (none of the project files is ignored by the type hinter). <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a> (Future work will add strict type hinting to the core API)</li>
<li>Action masking added to the taxi environment (no version bump due to being backwards compatible) <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>The <code>Box</code> space shape inference is allows <code>high</code> and <code>low</code> scalars to be automatically set to <code>(1,)</code> shape. Minor changes to identifying scalars. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Added option support in classic control environment to modify the bounds on the initial random state of the environment <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/psc-g/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/psc-g">@psc-g</a></li>
<li>The <code>RecordVideo</code> wrapper is becoming deprecated with no support for <code>TextEncoder</code> with the new render API. The plan is to replace <code>RecordVideo</code> with a single function that will receive a list of frames from an environment and automatically render them as a video using <code>MoviePy</code>.  <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/johnMinelli/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/johnMinelli">@johnMinelli</a></li>
<li>The gym <code>py.Dockerfile</code> is optimised from 2Gb to 1.5Gb through a number of optimisations <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/TheDen/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/TheDen">@TheDen</a></li>
</ul>

## 0.24.1

*Released on 2022-06-07 - [GitHub](https://github.com/openai/gym/releases/tag/0.24.1) - [PyPI](https://pypi.org/project/gym/0.24.1/)*

<p>This is a bug fix release for version 0.24.0</p>
<p>Bugs fixed:</p>
<ul>
<li>Replaced the environment checker introduced in V24, such that the environment checker will not call <code>step</code> and <code>reset</code> during make. This new version is a wrapper that will observe the data that <code>step</code> and <code>reset</code> returns on their first call and check the data against the environment checker. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Fixed MuJoCo v4 arguments key callback, closing the environment in renderer and the mujoco_rendering close method. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/rodrigodelazcano/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/rodrigodelazcano">@rodrigodelazcano</a></li>
<li>Removed redundant warning in registration <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
<li>Removed maths operations from MuJoCo xml files <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/quagla/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/quagla">@quagla</a></li>
<li>Added support for unpickling legacy <code>spaces.Box</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Fixed mujoco environment action and observation space docstring tables <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Disable wrappers from accessing <code>_np_random</code> property and <code>np_random</code> is now forwarded to environments <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Rewrite setup.py to add a "testing" meta dependency group <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Fixed docstring in <code>rescale_action</code> wrapper <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/gianlucadecola/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/gianlucadecola">@gianlucadecola</a></li>
</ul>

## 0.24.0

*Released on 2022-05-25 - [GitHub](https://github.com/openai/gym/releases/tag/0.24.0) - [PyPI](https://pypi.org/project/gym/0.24.0/)*

<p>Major changes</p>
<ul>
<li>Added v4 mujoco environments that use the new <a href="https://github.com/deepmind/mujoco">deepmind mujoco</a> 2.2.0 module.<br>
This can be installed through <code>pip install gym[mujoco]</code> with the old bindings still being<br>
available using the <code>v3</code> environments and <code>pip install gym[mujoco-py]</code>.<br>
These new <code>v4</code> environment should have the same training curves as <code>v3</code>. For the Ant, we found that there was a<br>
contact parameter that was not applied in <code>v3</code> that can enabled in <code>v4</code> however was found to produce significantly<br>
worse performance <a href="https://github.com/openai/gym/pull/2762#issuecomment-1135362092" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/2762/hovercard">see comment</a> for more details. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/rodrigodelazcano/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/rodrigodelazcano">@rodrigodelazcano</a></li>
<li>The vector environment step <code>info</code> API has been changes to allow hardware acceleration in the future.<br>
See <a href="https://github.com/openai/gym/pull/2773" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/2773/hovercard">this PR</a> for the modified <code>info</code> style that now uses dictionaries instead of a list of environment info.<br>
If you still wish to use the list info style, then use the <code>VectorListInfo</code> wrapper. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/gianlucadecola/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/gianlucadecola">@gianlucadecola</a></li>
<li>On <code>gym.make</code>, the gym <code>env_checker</code> is run that includes calling the environment <code>reset</code> and <code>step</code> to check if the<br>
environment is compliant to the gym API. To disable this feature, run <code>gym.make(..., disable_env_checker=True)</code>. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
<li>Re-added <code>gym.make("MODULE:ENV")</code> import style that was accidentally removed in v0.22 <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/arjun-kg/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/arjun-kg">@arjun-kg</a></li>
<li><code>Env.render</code> is now order enforced such that <code>Env.reset</code> is required before <code>Env.render</code> is called. If this a required<br>
feature then set the <code>OrderEnforcer</code> wrapper <code>disable_render_order_enforcing=True</code>. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Added wind and turbulence to the Lunar Lander environment, this is by default turned off,<br>
use the <code>wind_power</code> and <code>turbulence</code> parameter. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/virgilt/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/virgilt">@virgilt</a></li>
<li>Improved the <code>play</code> function to allows multiple keyboard letter to pass instead of ascii value <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Markus28/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Markus28">@Markus28</a></li>
<li>Added google style pydoc strings for most of the repositories <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Markus28/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Markus28">@Markus28</a></li>
<li>Added discrete car racing environment version through <code>gym.make("CarRacing-v1", continuous=False)</code></li>
<li>Pygame is now an optional module for box2d and classic control environments that is only necessary for rendering.<br>
Therefore, install pygame using <code>pip install gym[box2d]</code> or <code>pip install gym[classic_control]</code> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/gianlucadecola/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/gianlucadecola">@gianlucadecola</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
<li>Fixed bug in batch spaces (used in VectorEnv) such that the original space's seed was ignored <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Added <code>AutoResetWrapper</code> that automatically calls <code>Env.reset</code> when <code>Env.step</code> done is True <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/balisujohn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/balisujohn">@balisujohn</a></li>
</ul>
<p>Minor changes</p>
<ul>
<li>BipedalWalker and LunarLander's observation spaces have non-infinite upper and lower bounds. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jjshoots/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jjshoots">@jjshoots</a></li>
<li>Bumped the ALE-py version to <code>0.7.5</code></li>
<li>Improved the performance of car racing through not rendering polygons off screen @andrewtanJS</li>
<li>Fixed turn indicators that were black not red/white in Car racing <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jjshoots/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jjshoots">@jjshoots</a></li>
<li>Bug fixes for <code>VecEnvWrapper</code> to forward method calls to the environment <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/arjun-kg/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/arjun-kg">@arjun-kg</a></li>
<li>Removed unnecessary try except on Box2d such that if <code>Box2d</code> is not installed correctly then a more helpful error is show <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Simplified the <code>gym.registry</code> backend <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a></li>
<li>Re-added python 3.6 support through backports of python 3.7+ modules. This is not tested or compatible with the mujoco environments. <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
</ul>

## 0.23.1

*Released on 2022-03-11 - [GitHub](https://github.com/openai/gym/releases/tag/0.23.1) - [PyPI](https://pypi.org/project/gym/0.23.1/)*

<p>This release contains a few small bug fixes and no breaking changes.</p>
<ul>
<li>Make <code>VideoRecorder</code> backward-compatible to <code>gym&lt;0.23</code> by <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vwxyzjn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vwxyzjn">@vwxyzjn</a> in <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1164064529" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/2678" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/2678/hovercard" href="https://github.com/openai/gym/pull/2678">#2678</a></li>
<li>Fix issues with pygame event handling (which should fix support on windows and in jupyter notebooks) by @andrewtanJS in <a class="issue-link js-issue-link" data-error-text="Failed to load title" data-id="1166600972" data-permission-text="Title is private" data-url="https://github.com/openai/gym/issues/2684" data-hovercard-type="pull_request" data-hovercard-url="/openai/gym/pull/2684/hovercard" href="https://github.com/openai/gym/pull/2684">#2684</a></li>
<li>Add py.typed to package_data by <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/micimize/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/micimize">@micimize</a> in <a href="https://github.com/openai/gym/p">https://github.com/openai/gym/p</a></li>
<li>Fixes around 1500 warnings in CI <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/pseudo-rnd-thoughts/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/pseudo-rnd-thoughts">@pseudo-rnd-thoughts</a></li>
<li>Deprecation warnings correctly display now <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vwxyzjn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vwxyzjn">@vwxyzjn</a></li>
<li>Fix removing striker and thrower <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RushivArora/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RushivArora">@RushivArora</a></li>
<li>Fix small dependency warning errorr <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/ZhiqingXiao/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/ZhiqingXiao">@ZhiqingXiao</a></li>
</ul>

## 0.23.0

*Released on 2022-03-04 - [GitHub](https://github.com/openai/gym/releases/tag/0.23.0) - [PyPI](https://pypi.org/project/gym/0.23.0/)*

<p>This release contains many bug fixes and a few small changes.</p>
<p>Breaking changes:</p>
<ul>
<li>Standardized render metadata variables ahead of render breaking change <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/trigaten/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/trigaten">@trigaten</a></li>
<li>Removed deprecated monitor wrapper and associated dead code <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/gianlucadecola/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/gianlucadecola">@gianlucadecola</a></li>
<li>Unused striker and thrower MuJoCo envs moved to <a href="https://github.com/RushivArora/Gym-Mujoco-Archive">https://github.com/RushivArora/Gym-Mujoco-Archive</a> <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RushivArora/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RushivArora">@RushivArora</a></li>
</ul>
<p>Many minor bug fixes (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vwxyzjn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vwxyzjn">@vwxyzjn</a> , <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/RedTachyon/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/RedTachyon">@RedTachyon</a> , <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/rusu24edward/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/rusu24edward">@rusu24edward</a> , <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/Markus28/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/Markus28">@Markus28</a> , @dsctt , @andrewtanJS , <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/tristandeleu/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/tristandeleu">@tristandeleu</a> , <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/duburcqa/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/duburcqa">@duburcqa</a>)</p>

## 0.22.0

*Released on 2022-02-17 - [GitHub](https://github.com/openai/gym/releases/tag/0.22.0) - [PyPI](https://pypi.org/project/gym/0.22.0/)*

<h1>v0.22 Release Notes</h1>
<p>This release represents the largest set of changes ever to Gym, and represents a huge step towards the plans for 1.0 outlined here: <a href="https://github.com/openai/gym/pull/2524" data-hovercard-type="issue" data-hovercard-url="/openai/gym/issues/2524/hovercard">#2524</a></p>
<p>Gym now has a new comprehensive documentation site: <a href="https://www.gymlibrary.ml/" rel="nofollow">https://www.gymlibrary.ml/</a> !</p>
<h2>API changes</h2>
<ul>
<li>
<p><code>Env.reset</code> now accepts three new arguments:</p>
</li>
<li>
<p><code>options</code>: Usable for things like controlling curriculum learning without reinitializing the environment, which can be expensive (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</p>
</li>
<li>
<p><code>seed</code>: Environment seeds can be passed to this reset argument in the future. The old <code>.seed()</code> method is being deprecated in favor of this, though it will continue to function as before until the 1.0 release for backwards compatibility purposes (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</p>
</li>
<li>
<p><code>return_info</code>: when set to <code>True</code>, reset will return obs, info. This currently defaults to <code>False</code>, but will become the default behavior in Gym 1.0 (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</p>
</li>
<li>
<p>Environment names no longer require a version during registration and will suggest intelligent similar names (<a href="https://github.com/kir0ul">@kir0ul</a>, <a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</p>
</li>
<li>
<p>Vector environments now support terminal_observation in <code>info</code> and support batch action spaces (<a href="https://github.com/vwxyzjn">@vwxyzjn</a>, <a href="https://github.com/tristandeleu">@tristandeleu</a>)</p>
</li>
</ul>
<h2>Environment changes</h2>
<ul>
<li>The blackjack and frozen lake toy_text environments now have nice graphical rendering using PyGame (<a href="https://github.com/1b15">@1b15</a>)</li>
<li>Moved robotics environments to gym-robotics package (<a href="https://github.com/seungjaeryanlee">@seungjaeryanlee</a>, <a href="https://github.com/Rohan138">@Rohan138</a>, <a href="https://github.com/vwxyzjn">@vwxyzjn</a>) (per discussion in <a href="https://github.com/openai/gym/pull/2456" data-hovercard-type="issue" data-hovercard-url="/openai/gym/issues/2456/hovercard">#2456</a> (comment))</li>
<li>The bipedal walker and lunar lander environments were consolidated into one class (<a href="https://github.com/andrewtanJS">@andrewtanJS</a>)</li>
<li>Atari environments now use standard seeding API (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li>Fixed large bug fixes in car_racing box2d environment, bumped version (<a href="https://github.com/carlosluis">@carlosluis</a>, <a href="https://github.com/araffin">@araffin</a>)</li>
<li>Refactored all box2d and classic_control environments to use PyGame instead of Pyglet as issues with pyglet has been one of the most frequent sources of GitHub issues over the life of the gym project (<a href="https://github.com/andrewtanJS">@andrewtanJS</a>)</li>
</ul>
<h2>Other changes</h2>
<ul>
<li>Removed DiscreteEnv class, built in environments no longer use it (<a href="https://github.com/carlosluis">@carlosluis</a>)</li>
<li>Large numbers of type hints added (<a href="https://github.com/ikamensh">@ikamensh</a>, <a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>Python 3.10 support</li>
<li>Tons of additional code refactoring, cleanup, error message improvements and small bug fixes (<a href="https://github.com/vwxyzjn">@vwxyzjn</a>, <a href="https://github.com/Markus28">@Markus28</a>, <a href="https://github.com/RushivArora">@RushivArora</a>, <a href="https://github.com/jjshoots">@jjshoots</a>, <a href="https://github.com/XuehaiPan">@XuehaiPan</a>, <a href="https://github.com/Rohan138">@Rohan138</a>, <a href="https://github.com/JesseFarebro">@JesseFarebro</a>, <a href="https://github.com/Ericonaldo">@Ericonaldo</a>, <a href="https://github.com/AdilZouitine">@AdilZouitine</a>, <a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>All environment files now have dramatically improved readmes at the top (that the documentation website automatically pulls from)</li>
<li>As part of the seeding changes, Gym's RNG has been modified to use the <code>np.random.Generator</code> as the RandomState API has been deprecated. The methods <code>randint</code>, <code>rand</code>, <code>randn</code> are replaced by <code>integers</code>, <code>random</code> and <code>standard_normal</code> respectively. As a consequence, the random number generator has changed from <code>MT19937</code> to <code>PCG64</code>.</li>
</ul>
<p><strong>Full Changelog</strong>: <a class="commit-link" href="https://github.com/openai/gym/compare/v0.21.0...0.22.0"><tt>v0.21.0...0.22.0</tt></a></p>

## v0.21.0

*Released on 2021-10-02 - [GitHub](https://github.com/openai/gym/releases/tag/v0.21.0) - [PyPI](https://pypi.org/project/gym/v0.21.0/)*

<h1>v0.21.0 Release Notes</h1>
<ul>
<li>The old Atari entry point that was broken with the last release and the upgrade to ALE-Py is fixed (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li>Atari environments now give much clearer error messages and warnings (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li>A new plugin system to enable an easier inclusion of third party environments has been added (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li>Atari environments now use the new plugin system to prevent clobbered names and other issues (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li><code>pip install gym[atari]</code> no longer distributes Atari ROMs that the ALE (the Atari emulator used) needs to run the various games. The easiest way to install ROMs into the ALE has been to use <a href="https://github.com/Farama-Foundation/AutoROM">AutoROM</a>. Gym now has a hook to AutoROM for easier CI automation so that using <code>pip install gym[accept-rom-license]</code> calls AutoROM to add ROMs to the ALE. You can install the entire suite with the shorthand <code>gym[atari, accept-rom-license]</code>. Note that as described in the name name, by installing <code>gym[accept-rom-license]</code> you are confirming that you have the relevant license to install the ROMs. (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li>An accidental breaking change when loading saved policies trained on old versions of Gym with environments using the box action space have been fixed. (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>Pendulum has had a minor fix to it's physics logic made and the version has been bumped to v1 (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>Tests have been refactored into an orderly manner (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>Dict spaces now have standard dict helper methods (<a href="https://github.com/Rohan138">@Rohan138</a>)</li>
<li>Environment properties are now forwarded to the wrapper (<a href="https://github.com/tristandeleu">@tristandeleu</a>)</li>
<li>Gym now properly enforces calling reset before stepping for the first time (<a href="https://github.com/ahmedo42">@ahmedo42</a>)</li>
<li>Proper piping of error messages to stderr (<a href="https://github.com/XuehaiPan">@XuehaiPan</a>)</li>
<li>Fix video saving issues (<a href="https://github.com/zlig">@zlig</a>)</li>
</ul>
<p>Also, Gym is compiling a list of third party environments to into the new documentation website we're working on. Please submit PRs for ones that are missing: <a href="https://github.com/openai/gym/blob/master/docs/third_party_environments.md">https://github.com/openai/gym/blob/master/docs/third_party_environments.md</a></p>
<p><strong>Full Changelog</strong>: <a class="commit-link" href="https://github.com/openai/gym/compare/v0.20.0...v0.21.0"><tt>v0.20.0...v0.21.0</tt></a></p>

## v0.20.0

*Released on 2021-09-14 - [GitHub](https://github.com/openai/gym/releases/tag/v0.20.0) - [PyPI](https://pypi.org/project/gym/v0.20.0/)*

<h1>v0.20.0 Release Notes</h1>
<h2>Major Change</h2>
<ul>
<li>Replaced Atari-Py dependency with ALE-Py and bumped all versions. This is a massive upgrade with many changes, please see the <a href="https://brosa.ca/blog/ale-release-v0.7" rel="nofollow">full explainer</a> (<a href="https://github.com/JesseFarebro">@JesseFarebro</a>)</li>
<li>Note that ALE-Py does not include ROMs. You can install ROMs in two lines of bash with <code>AutoROM</code> though (<code>pip3 install autorom and then autorom</code>), see <a href="https://github.com/PettingZoo-Team/AutoROM">https://github.com/PettingZoo-Team/AutoROM</a>. This is the recommended approach for CI, etc.</li>
</ul>
<h2>Breaking changes and new features:</h2>
<ul>
<li>Add <code>RecordVideo</code> wrapper, deprecate <code>monitor</code> wrapper in favor of it and <code>RecordEpisodeStatistics</code> wrapper (<a href="https://github.com/vwxyzjn">@vwxyzjn</a>)</li>
<li>Dependencies used outside of environments (e.g. for wrappers) are now in <code>gym[other]</code> (<a href="https://github.com/jkterry1">@jkterry1</a>)</li>
<li>Moved algorithmic and unused toy-text envs (guessing game, hotter colder, nchain, roulette, kellycoinflip) to third party repos (<a href="https://github.com/jkterry1">@jkterry1</a>, <a href="https://github.com/Rohan138">@Rohan138</a>)</li>
<li>Fixed flatten utility and flatdim in MultiDiscrete space (<a href="https://github.com/tristandeleu">@tristandeleu</a>)</li>
<li>Add <code>_\_setitem_\_</code> to dict space (<a href="https://github.com/jfpettit">@jfpettit</a>)</li>
<li>Large fixes to <code>.contains</code> method for box space (<a href="https://github.com/FirefoxMetzger">@FirefoxMetzger</a>)</li>
<li>Made blackjack environment properly comply with Barto and Sutton book standard, bumped to v1 (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>Added <code>NormalizeObservation</code> and <code>NormalizeReward</code> wrappers (<a href="https://github.com/vwxyzjn">@vwxyzjn</a>)</li>
<li>Add <code>_\_getitem_\_</code> and <code>_\_len_\_</code> to MultiDiscrete space (<a href="https://github.com/XuehaiPan">@XuehaiPan</a>)</li>
<li>Changed <code>.shape</code> to be a property of box space to prevent unexpected behaviors (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
</ul>
<h2>Bug fixes and upgrades</h2>
<ul>
<li>Video recorder gracefully handles closing (<a href="https://github.com/XuehaiPan">@XuehaiPan</a>)</li>
<li>Remaining unnecessary dependencies in setup.py are resolved (<a href="https://github.com/jkterry1">@jkterry1</a>)</li>
<li>Minor acrobot performance improvements (<a href="https://github.com/TuckerBMorgan">@TuckerBMorgan</a>)</li>
<li>Pendulum properly renders when 0 force is sent (<a href="https://github.com/Olimoyo">@Olimoyo</a>)</li>
<li>Make observations dtypes be consistent with observation space dtypes for all classic control envs and bipedal-walker (<a href="https://github.com/RedTachyon">@RedTachyon</a>)</li>
<li>Removed unused and long deprecated features in registration (<a href="https://github.com/Rohan138">@Rohan138</a>)</li>
<li>Framestack wrapper now inherits from obswrapper (<a href="https://github.com/jfpettit">@jfpettit</a>)</li>
<li>Seed method for <code>spaces.Tuple</code> and <code>spaces.Dict</code> now properly function, are fully stochastic, are fully featured and behave in the expected manner (<a href="https://github.com/XuehaiPan">@XuehaiPan</a>, <a href="https://github.com/RaghuSpaceRajan">@RaghuSpaceRajan</a>)</li>
<li>Replace <code>time()</code> with <code>perf_counter()</code> for better measurements of short duration (<a href="https://github.com/zuoxingdong">@zuoxingdong</a>)</li>
</ul>
<p><strong>Full Changelog</strong>: <a class="commit-link" href="https://github.com/openai/gym/compare/0.19.0...v0.20.0"><tt>0.19.0...v0.20.0</tt></a></p>

## 0.19.0

*Released on 2021-08-13 - [GitHub](https://github.com/openai/gym/releases/tag/0.19.0) - [PyPI](https://pypi.org/project/gym/0.19.0/)*

<p>Gym 0.19.0 is a large maintenance release, and the first since <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jkterry1/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jkterry1">@jkterry1</a> became the maintainer. There should be no breaking changes in this release.</p>
<p>New features:</p>
<ul>
<li>Added custom datatype argument to multidiscrete space (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/m-orsini/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/m-orsini">@m-orsini</a>)</li>
<li>API compliance test added based on SB3 and PettingZoo tests (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/amtamasi/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/amtamasi">@amtamasi</a>)</li>
<li>RecordEpisodeStatics works with VectorEnv (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/vwxyzjn/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/vwxyzjn">@vwxyzjn</a>)</li>
</ul>
<p>Bug fixes:</p>
<ul>
<li>Removed unused dependencies, removed unnescesary dependency version requirements that caused installation issues on newer machines, added full requirements.txt and moved general dependencies to extras. Notably, "toy_text" is not a used extra. atari-py is now pegged to a precise working version pending the switch to ale-py (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jkterry1/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jkterry1">@jkterry1</a>)</li>
<li>Bug fixes to rewards in FrozenLake and FrozenLake8x8; versions bumped to v1 (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/ZhiqingXiao/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/ZhiqingXiao">@ZhiqingXiao</a>)<br>
-Removed remaining numpy depreciation warnings (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/super-pirata/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/super-pirata">@super-pirata</a>)</li>
<li>Fixes to video recording (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/mahiuchun/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/mahiuchun">@mahiuchun</a>, <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/zlig/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/zlig">@zlig</a>)</li>
<li>EZ pickle argument fixes (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/zzyunzhi/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/zzyunzhi">@zzyunzhi</a>, @Indoril007)</li>
<li>Other very minor (nonbreaking) fixes</li>
</ul>
<p>Other:</p>
<ul>
<li>Removed small bits of dead code (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jkterry1/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jkterry1">@jkterry1</a>)</li>
<li>Numerous typo, CI and documentation fixes (mostly <a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/cclauss/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/cclauss">@cclauss</a>)</li>
<li>New readme and updated third party env list (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/jkterry1/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/jkterry1">@jkterry1</a>)</li>
<li>Code is now all flake8 compliant through black (<a class="user-mention notranslate" data-hovercard-type="user" data-hovercard-url="/users/cclauss/hovercard" data-octo-click="hovercard-link-click" data-octo-dimensions="link_type:self" href="https://github.com/cclauss">@cclauss</a>)</li>
</ul>

## 0.12.5

*Released on 2019-05-29 - [GitHub](https://github.com/openai/gym/releases/tag/0.12.5) - [PyPI](https://pypi.org/project/gym/0.12.5/)*

## v0.9.6

*Released on 2018-02-01 - [GitHub](https://github.com/openai/gym/releases/tag/v0.9.6) - [PyPI](https://pypi.org/project/gym/v0.9.6/)*

<ul>
<li>Now your <code>Env</code> and <code>Wrapper</code> subclasses should define <code>step</code>, <code>reset</code>, <code>render</code>, <code>close</code>, <code>seed</code> rather than underscored method names.</li>
<li>Removed the <code>board_game</code>, <code>debugging</code>, <code>safety</code>, <code>parameter_tuning</code> environments since they're not being maintained by us at OpenAI. We encourage authors and users to create new repositories for these environments.</li>
<li>Changed <code>MultiDiscrete</code> action space to range from <code>[0, ..., n-1]</code> rather than <code>[a, ..., b-1]</code>.</li>
<li>No more <code>render(close=True)</code>, use env-specific methods to close the rendering.</li>
<li>Removed <code>scoreboard</code> directory, since site doesn't exist anymore.</li>
<li>Moved <code>gym/monitoring</code> to <code>gym/wrappers/monitoring</code></li>
<li>Add <code>dtype</code> to <code>Space</code>.</li>
<li>Not using python's built-in module anymore, using <code>gym.logger</code></li>
</ul>

## v0.9.5

*Released on 2018-01-26 - [GitHub](https://github.com/openai/gym/releases/tag/v0.9.5) - [PyPI](https://pypi.org/project/gym/v0.9.5/)*

## v0.7.4

*Released on 2017-03-05 - [GitHub](https://github.com/openai/gym/releases/tag/v0.7.4) - [PyPI](https://pypi.org/project/gym/v0.7.4/)*

## v0.7.3

*Released on 2017-02-01 - [GitHub](https://github.com/openai/gym/releases/tag/v0.7.3) - [PyPI](https://pypi.org/project/gym/v0.7.3/)*
