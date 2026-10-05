# OpenDecider Focus privacy policy

*Effective 5 October 2026. Applies to OpenDecider Focus, the Chrome extension, from version 0.8.1.*

OpenDecider Focus decides on your computer. It reads the titles of the videos on the YouTube pages it filters, and the
text you ask it to check, with opendecider-nano running in your browser. That text is never sent to us or to anyone
else. There is no server, no account, no analytics and no advertising. The extension is made by Manjunath Janardhan,
and its code is public: [github.com/manjunathshiva/opendecider](https://github.com/manjunathshiva/opendecider/tree/main/extension).

## What it reads, and why

| what | when | why | where it goes |
|---|---|---|---|
| The title, channel name and video id of each video on YouTube's home page, search results and the list beside a video (website content) | while the filter is on and you have downloaded the model | to decide whether to show or hide the video | nowhere: the model runs in your browser |
| Shorts, recognised by their links (`/shorts/`) by the extension's stylesheet: no code reads them, and nothing is kept | while the filter is on and **Hide Shorts** is ticked, also before the model is downloaded | to hide Shorts | nowhere |
| The text you select on a page and check with **Check with OpenDecider guard** (website content) | only when you choose that menu item | to check it for prompt injection | nowhere: the model runs in your browser |
| Your settings: the kinds of video to hide, your own rule, how strict to be, Shorts | when you change them | to apply them | Chrome's synced storage, so they follow your Chrome profile if you use Chrome Sync |

## What it keeps on your computer

- **Answers already computed:** for up to 5,000 videos, the video id and the model's probabilities (never the title),
  so a video is not judged twice. They are kept in the extension's local storage.
- **The model's files:** opendecider-nano (569–755 MiB), kept in the browser's cache after the first download.
- **The text you check:** kept in Chrome's session storage (memory, never written to disk) until the guard window
  reads it, then deleted from there. The guard window and the model's page hold it in memory while they check and show
  it, and it is gone when the window closes. If the window closes before reading it, it stays in session storage until
  Chrome closes.

Uninstalling the extension deletes all of this. You can also clear it from Chrome's site data for the extension.

## What it downloads

Once you press **Download the model**, the extension downloads the model's files from Hugging Face
(`huggingface.co`), at a revision pinned in the extension, and checks each file's SHA-256 before using it. Like any
download, the request carries your IP address and browser details to Hugging Face, under
[Hugging Face's privacy policy](https://huggingface.co/privacy); it carries nothing about what you watch. The
extension makes no other network request: its pages' Content Security Policy allows only Hugging Face.

## What it never does

- It never sends what you watch, search or select to us or to anyone else.
- It never sells or transfers your data, and never uses it for advertising, credit or lending decisions, or anything
  other than filtering your feed and checking text for you.
- It reads no page other than YouTube, except the text you choose to check.

The extension's handling of user data complies with the
[Chrome Web Store User Data Policy](https://developer.chrome.com/docs/webstore/program-policies/policies), including
its Limited Use requirements.

## Changes and contact

A change to this policy is published on this page with a new effective date, and in the
[changelog](https://github.com/manjunathshiva/opendecider/blob/main/CHANGELOG.md). Questions:
[open an issue](https://github.com/manjunathshiva/opendecider/issues); security reports: see
[SECURITY.md](https://github.com/manjunathshiva/opendecider/blob/main/SECURITY.md).
