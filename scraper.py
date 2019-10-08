#!/usr/bin/env python3


import praw
import pandas as pd
import datetime as dt

from secret import personal_use_script, client_secret, password

reddit = praw.Reddit(
    client_id=personal_use_script,
    client_secret=client_secret,
    user_agent="cursed_scraper 1",
    username="epicepee",
    password=password,
)

sub = reddit.subreddit("cursedimages")

top = sub.top(limit=2000)

posts = {}
# for post in sub.top(limit=1000):
for post in (sub.random() for _ in range(10000)):
    filename = post.title + "_" + post.id
    print(filename)
    url = post.url
    posts[filename] = url

print(len(posts))
