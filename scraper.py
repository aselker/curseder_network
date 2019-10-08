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

top = sub.top(limit=1000)

for post in top:
    print(post.title, post.id)
