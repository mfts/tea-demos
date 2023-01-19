#!/usr/bin/env python3

"""
courtesy of https://github.com/stoooops

This is a pre-commit-msg hook that uses the OpenAI API to get a suggested commit message
for the diff that is about to be committed.

To use this hook, you will need to:
- Set the OPENAI_API_KEY environment variable to your OpenAI API key.

To install this hook, follow these steps:
1. Place this file in the `.git/hooks` directory in your Git repository.
2. Make sure the file is executable. You can do this by running the following command:
   `chmod +x .git/hooks/pre-commit-msg`

This hook is invoked by Git just before the commit message editor is launched,
and it is passed the name of the file that holds the commit message.
The hook should edit this file in place and then exit.
If the hook exits non-zero, Git aborts the commit process.
"""

import http.client
import json
import os
import subprocess
import sys
from urllib.parse import urlparse


class OpenAIApiException(Exception):
    """
    An exception that is raised when the OpenAI API returns an error.
    """

    pass

class OpenAIBadCompletionException(OpenAIApiException):
    """
    An exception that is raised when the OpenAI API returns a bad completion.
    """

    pass


class OpenAIApiClient:
    """
    A simple wrapper around the OpenAI API
    """

    def __init__(self, api_key=None):
        """
        Constructor for the OpenApiClient class.

        Parameters:
        api_key (str, optional): The OpenAI API key to be used. If not provided, the value of the
            OPENAI_API_KEY environment variable will be used.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.url: str = "https://api.openai.com/v1/completions"

    def _create_request_data(self, prompt, model, max_tokens, temperature):
        """
        Create the request data payload to be sent to the OpenAI API.

        Parameters:
        prompt (str): The prompt to complete.
        model (str): The name of the model to use for generating the completion.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature to use for generating the completion.

        Returns:
        dict: The request data payload.
        """
        return {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1,
        }

    def _create_headers(self):
        """
        Create the headers for the request to the OpenAI API.

        Returns:
        dict: The headers for the request.
        """
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}

    def _send_request(self, data, headers):
        """
        Send a POST request to the OpenAI API.

        Parameters:
        data (dict): The request data payload.
        headers (dict): The headers for the request.

        Returns:
        dict: The response data.
        """
        # parse the URL
        url = urlparse(self.url)

        # create the HTTPS connection to the API
        conn = http.client.HTTPSConnection(host=url.hostname)

        # encode the data payload as a JSON string
        data = json.dumps(data).encode("utf-8")

        # send the POST request to the API
        conn.request("POST", url.path, body=data, headers=headers)

        # get the response from the API
        response = conn.getresponse()

        # read the response data
        response_data = response.read()

        # parse the response data as JSON
        return json.loads(response_data)

    def _check_response(self, response):
        """
        Check if the response from the OpenAI API is valid.

        Parameters:
        response (dict): The response data.

        Returns:
        bool: True if the response is valid, False otherwise.
        """
        if "choices" not in response:
            if "error" in response:
                error = response["error"]["message"] if "message" in response["error"] else response["error"]
                print(f"\033[01;31m{error}\033[0;0m", file=sys.stderr)
            else:
                print(f"\033[01;31m{json.dumps(response, indent=2)}\033[0;0m", file=sys.stderr)
            return False
        return True

    def _get_completion(self, response):
        """
        Get the completion from the response data.

        Parameters:
        response (dict): The response data.

        Returns:
        str: The completion.
        """
        return response["choices"][0]["text"].strip()

    def get_suggested_commit_message(self, prompt, model="text-davinci-003", max_tokens=256, temperature=0.85) -> str:
        """
        Get a completion from the OpenAI API.

        Parameters:
        prompt (str): The prompt to complete.
        model (str, optional): The name of the model to use for generating the completion.
            Defaults to "text-davinci-003".
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 64.
        temperature (float, optional): The temperature to use for generating the completion.
            Defaults to 0.9.

        Returns:
        str: The completion generated by the OpenAI API.
        """
        request_data = self._create_request_data(prompt, model, max_tokens, temperature)
        headers = self._create_headers()
        response = self._send_request(request_data, headers)
        if not self._check_response(response):
            raise OpenAIApiException("OpenAI API returned an invalid response")
        completion = self._get_completion(response)
        with open(".prompt", "a") as f:
            f.write(completion)

        # first two characters should be "🤖 "
        if not completion.startswith("🤖 "):
            raise OpenAIBadCompletionException(f"OpenAI API commit message incorrect header symbol:\n{completion}")

        # should not begin with "🤖 Update"
        if completion.startswith("🤖 Update"):
            raise OpenAIBadCompletionException(f"OpenAI API bad commit message:\n'{completion}'")

        return completion

def get_prompt(model: str, status_text: str, diff_text: str) -> str:
    return f"""
I want you to act as a technical writer for software engineers, your primary responsibility
is to write clear and concise commit messages for code changes. Your job is to communicate
the purpose and impact of code changes to other members of the development team. It is
important to provide context and details about the changes made, but avoid including
personal opinions or subjective evaluations in your messages.

A good commit message has the following characteristics:
- It is concise and accurately describes the changes made in the commit.
- It is written in the imperative mood and begins with a verb
- It explains why the change was made, rather than how it was made.
- It includes a signature at the end of the message.

Here is an example of a good commit message:
```
🤖 Fix password bug in login form

The login form was submitting even if the password field was empty.
This commit fixes the bug by checking that the password field is not
empty before allowing the form to be submitted.


(commit message written by OpenAI {model})
```

Here is an example of a good commit message:
```text
🤖 Add prettier and eslint dependencies

This commit updates the package.json file to include the latest
dependencies for prettier and eslint.

prettier is a code formatter that automatically formats code to
conform to a consistent style. It is configured to use the
recommended settings for the JavaScript Standard Style.

eslint is a linter that checks for common errors and code smells.
It is configured to use the recommended settings for the
JavaScript Standard Style.

(commit message written by OpenAI {model})
```

Some other tips for writing good commit messages:
- Separate subject from body with a blank line
- Keep the subject line (the first line) to 50 characters or less
- Use the body of the message to explain the details of the commit, if necessary
- Wrap the body at 72 characters
- The first two characters should be "🤖 " to indicate that the commit message was written by an AI model
- Do not start with the word "Update" or anything implied by a commit

At the end of the commit message, add a signature:
```text
(commit message written by OpenAI {model})
```

Your first task is to review staged changes and suggest a clear and concise commit message for the latest code update to ensure they meet the required standards described above for clarity and conciseness.

Now please, write a commit message for the following patch, starting with "🤖 ".:

Files changed:
```
// git status -s
{status_text}
```

Files diff:
```diff
// git diff --cached --no-color --no-ext-diff --unified=0 --no-prefix
{diff_text}
```
Lastly, avoid starting the message with "🤖 Update". Instead, choose a unique and stylish first line in the imperative tense that concisely describes the changes made in the commit. This line should be no more than 50 characters.

Now, please write a suggested commit message below that is clear, concise, and colorful, following the rules described above, beginning with "🤖 " and ending with the signature "(commit message written by OpenAI {model})":

Respond with the suggested commit message below starting with "🤖 " (unquoted).
"""


def check_abort() -> None:
    """
    Check if the commit message file is not empty or if the OPENAI_API_KEY environment variable is not set.

    If the commit message file is not empty, print a message and exit.
    If the OPENAI_API_KEY environment variable is not set, print an error message in red and exit.
    """
    # Check if the commit message file is not empty
    with open(sys.argv[1], "r") as f:
        if f.readline().strip():
            print("Commit message already exists, exiting")
            exit(0)

    # Check if the OPENAI_API_KEY environment variable is not set
    if "OPENAI_API_KEY" not in os.environ:
        # Print an error message in red
        print("\033[0;31mOpenAI suggestion failed: OPENAI_API_KEY not set\033[0m")
        exit(1)


def get_status_text() -> str:
    """
    Get the status text for the staged changes in the current Git repository.

    The `--short` option tells `git status` to output the status in a shorter format.
    The `--untracked-files=no` option tells `git status` to ignore untracked files.
    Together, these options limit the output of `git status` to only report files which are staged for commit.

    Returns:
    str: The status text for the staged changes in the current Git repository.
    """
    # Get the status text for the staged changes in the current Git repository
    result: subprocess.CompletedProcess = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if result.stderr:
        print("\033[0;31m", result.stderr, "\033[0m")
    if result.returncode != 0:
        raise Exception("git diff failed")

    return result.stdout



def get_diff_text(excluded=["package-lock.json", "yarn.lock"]) -> str:
    """
    Get the diff text for the staged changes in the current Git repository.

    Returns:
    str: The diff text for the staged changes in the current Git repository, with a maximum length of 10000 characters.
    """
    # Find the filenames of the staged changes in the current Git repository, excluding package-lock.json and yarn.lock
    # diff-filter=ACMRTUXB means: Added (A), Copied (C), Modified (M), Renamed (R), Changed (T), Updated but unmerged (U), eXisting (X), Broken (B)
    result: subprocess.CompletedProcess = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRTUXB"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if result.stderr:
        print("\033[0;31m", result.stderr, "\033[0m")
    if result.returncode != 0:
        raise Exception("git diff failed")

    # Get the diff text for the staged changes in the current Git repository
    staged_changes = [filename for filename in result.stdout.splitlines() if filename not in excluded]
    args = [
        "git",
        "diff",
        "--cached",
        "--no-color",
        "--no-ext-diff",
        "--unified=0",
        "--no-prefix",
    ] + staged_changes
    result: subprocess.CompletedProcess = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    )
    if result.stderr:
        print("\033[0;31m", result.stderr, "\033[0m")
    if result.returncode != 0:
        raise Exception("git diff failed")

    # the output may be too long so we will take the first 10000 characters
    return result.stdout[:10000]


def main() -> None:
    """
    Use the OpenAI API to get a suggested commit message for the diff that is about to be committed.
    """
    # Check if the commit should be aborted
    check_abort()

    # Get the status text and diff text for the staged changes in the current Git repository
    git_status_text: str = get_status_text()
    git_diff_text: str = get_diff_text()

    model: str = "text-davinci-003"

    # Get the prompt
    prompt: str = get_prompt(model=model, status_text=git_status_text, diff_text=git_diff_text)
    # save prompt to debug file
    with open(".prompt", "w") as f:
        f.write(prompt)

    # Get the suggested commit message
    print("Getting suggested commit message...")
    suggested_commit_message: str = OpenAIApiClient().get_suggested_commit_message(
        prompt=prompt, model=model
    )
    # delete the commit message file
    os.remove(sys.argv[1])

    # directly run gitf  commit -m "suggested_commit_message"
    # write commit message to file
    with open(sys.argv[1], "w") as f:
        f.write(suggested_commit_message)

    print()
    print(f"Wrote suggested commit message to {sys.argv[1]}")
    print()
    for line in suggested_commit_message.splitlines():
        # color code \033[0;90m
        print(f"> \033[0;90m{line}\033[0m")


if __name__ == "__main__":
    """
    Run the main function.
    """
    main()