#!/bin/bash

set -e


tea +charm.sh/gum gum format --<<EOMD
# setup a git hook for prepare-commit-msg

## what to expect from this script

* checks for \`.git\` repository
* fetches "prepare-commit-msg" script from remote 
* adds "prepare-commit-msg" in git hooks dir
* adds \`OPENAI_API_KEY\` to env
EOMD

tea +charm.sh/gum gum confirm "shall we continue?"


if ! test -d ".git";
# if no .git dir found, then ask to go to a git directory
then
  echo "no git directory found" >&2
  exit 1
fi

# clone prepare-commit-msg file to git_hook_path
GIT_HOOKS_PATH=$(tea +git-scm.org git rev-parse --git-path hooks)
tea +gnu.org/wget wget -O $GIT_HOOKS_PATH/prepare-commit-msg https://github.com/mfts/tea-demos/blob/feat/ai-commit-msg/ai-commit-msg/prepare-commit-msg.py

# make the file executable
chmod +x $GIT_HOOKS_PATH/prepare-commit-msg

# add python3 to the env via tea
tea +python.org^3 echo && source <(tea +python.org^3 --dry-run)

# add OPENAI_API_KEY to env
tea +charm.sh/gum gum format "finally, you need an \`OPENAI_API_KEY\` in your environment"
OPENAI_API_KEY=$(tea +charm.sh/gum gum input --prompt "what your \`OPENAI_API_KEY\`? ")

export OPENAI_API_KEY=$OPENAI_API_KEY

tea +charm.sh/gum gum format <<EOMD
# done!
Your git commits will now be generated by AI! 

EOMD
