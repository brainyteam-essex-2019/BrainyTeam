# BrainyTeam
Repository for the CE903 Module Assignment.

## Team Collaboration

We use git to do the collaboration around the codes.

For command line users, run:

```
git clone git@github.com:brainyteam-essex-2019/BrainyTeam.git
```

Or if you prefer https (or haven't got an ssh, e.g not configured windows system):
```
git clone https://github.com/brainyteam-essex-2019/BrainyTeam.git
```

For users preferring graphical user interface, you may like one of the provided tools below

* (Linux, Mac, Windows) [GitKraken](https://www.gitkraken.com/git-client)
* (Mac, Windows) [SourceTree](https://www.sourcetreeapp.com/)

### Working on your own branch

* To prevent damage to other's work, no one should work on `master` branch.
* When you have started to work on a new feature/bugfix/idea, use `git checkout -b <branchname>` to
create and switch to the new branch from the current workspace
(i.e: if you run `git checkout` on master, you start this branch from master, otherwise you start this branch from where you are)
* It is recommended that for each *card* in *Trello*, create a related branch for it

### Commiting the code

We strongly recommend you to manage the code using git even when not syncing with others.

* **Note**: In git, `commit` **ONLY** affects your local machine, and `fetch/rebase/pull/push` will sync with others
* **When to commit?**: Every time you think your progress is worth a simple sentence to describe, commit it.
    * If it worths several sentence, then the commit is too large, try deviding it to several smaller commits.
* **What to do before a commit?**: `Git status` or use the GUI tool to check
    * If you are on the expected branch
    * If there are unexpected changes in the workspace
* **How do I commit**:
    * Command line:
        * `git add <file>` to stage changes you want to include in this commit (related to the "sentence" describing the commit)
        * `git commit -m <message>` to describe what you did in this commit
    * GUI:
        * Check and stage the files you want to include in this commit
        * Commit the staged changes with a single line message describing what you did

### Sync with others

* If your current branch is never pushed to the remote, `git push -u origin <branch-name>`to push your branch to the remote and set track of it
* Before pushing your latest progress in a branch, run `git pull` to ensure that you are sync with the latest progress to the remote
* `git push` to push the latest commited progress in the current branch
    * Changes not included inside commits will not be pushed
* When finished working on that branch, create a pull request on github from the branch you are working on to the master branch, and assign it to either
Clyce or Jose, slack will automatically broadcast the pull request
* Anyone are free to review and comment on the pull request
* After all of the comments are resolved in the code review of the pull request, the branch will be merged into master