
# install gh
(type -p wget >/dev/null || (sudo apt update && sudo apt-get install wget -y)) \
	&& sudo mkdir -p -m 755 /etc/apt/keyrings \
        && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& sudo apt update \
	&& sudo apt install gh -y

git config --global user.email "adrienbufort@gmail.com"
git config --global user.name "Adrien B"

gh auth login
git clone https://github.com/meteolibre-dev/flashnet.git

cd flashnet/
pip install .

hf auth login

cd ..
mkdir dataset
cd dataset

# cli to download data
hf download meteolibre-dev/weather_mtg_world_lightning_128_0dot012_full --repo-type dataset --local-dir .

# if we want to use heavyball sudo apt-get -y install build-essential
apt install gcc # for heavyball / triton