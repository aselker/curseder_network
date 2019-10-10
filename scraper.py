#!/usr/bin/env python3


import praw
import pandas as pd
import datetime as dt
import sys
import urllib
import subprocess

from secret import personal_use_script, client_secret, password

reddit = praw.Reddit(
    client_id=personal_use_script,
    client_secret=client_secret,
    user_agent="cursed_scraper 1",
    username="epicepee",
    password=password,
)

sub = reddit.subreddit(sys.argv[1])

top = sub.top(limit=2000)

# for post in sub.top(limit=1000):
for post in (sub.random() for _ in range(10000)):
    if post:
        filename = post.title + "_" + post.id
        url = post.url
        if not (
            url.startswith("https://i.redd.it") or url.startswith("https://imgur.com")
        ):
            continue
        if url[-4:] in [".jpg", ".png"]:
            filename = filename + url[-4:]
        else:
            continue

        command = ["wget", url, "-O " + '"' + filename + '"']
        subprocess.call(command)
