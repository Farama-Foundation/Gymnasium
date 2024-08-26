##### Gymnasium Examples
###### 15-Sep-2024

Launch PyCharm | Open project | examples directory
<br />
IMPORTANT: Atari examples must based off Tag v0.29.1 in order to work  
```
git fetch --tags --prune --prune-tags
git fetch --all
git checkout v0.29.1
git checkout -b GymnasiumGameExamples
git push --set-upstream origin GymnasiumGameExamples
```
[Pre-Requisites](https://gymnasium.farama.org) | [ClassicControl](https://gymnasium.farama.org/environments/classic_control) | [ToyText](https://gymnasium.farama.org/environments/toy_text)
```
pip install -r docs/requirements.txt
pip install --upgrade pip
```
[Box2D](https://gymnasium.farama.org/environments/box2d)
```
pip install Box2D
```
[muJoCo](https://gymnasium.farama.org/environments/mujoco)
```
pip install mujoco==2.3.0
```
[Atari](https://gymnasium.farama.org/environments/atari)
```
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]
```