# Chrome extension

**OpenDecider Focus** filters your YouTube feed with opendecider-nano running on your computer. It hides the kinds of
video you choose, or the topics you name in your own words, and shows the rest. The titles it reads are never sent
anywhere: there is no server, no account and no API key, and after one download it works offline.

It also checks text for prompt injection: select text on any page, right-click it, and choose **Check with OpenDecider
guard**.

## Install

Until it is on the Chrome Web Store, install it from a release:

1. Download `opendecider-focus-<version>.zip` from the [latest release](https://github.com/manjunathshiva/opendecider/releases/latest)
   and unzip it.
2. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and pick the unzipped folder.
3. Click the OpenDecider Focus button in the toolbar and press **Download the model** (once: 755 MiB on a computer
   with a usable GPU, 569 MiB without one).

Each release's zip carries a build provenance attestation and a Sigstore signature: `gh attestation verify
opendecider-focus-<version>.zip --repo manjunathshiva/opendecider`. It needs Chrome 124 or later, on a computer
(Chrome on phones has no extensions).

## What it filters

The popup has three filters; a video is hidden when any of them says so.

- **Kinds of video.** Each video is sorted into one of 11 kinds: news, how-tos and recipes, science and tech, lectures
  and explainers, business, interviews and podcasts, music, gaming, comedy, celebrity and TV, vlogs. Tick the kinds to
  hide. **Focus** (the default) hides music, gaming, comedy, celebrity and TV, and vlogs.
- **Your own rule.** "Hide videos about crypto, celebrity gossip", or "Show only videos about cooking". The topics
  become one yes/no question about each video ("Is this video about crypto, celebrity gossip?"); write a whole question
  ending in `?` to ask it as written.
- **Shorts.** Hidden by their links, without the model.

**How sure before showing a video** moves the line between showing and hiding: towards "Show more", a video the model is
unsure about stays; towards "Hide more", it goes. It filters the home page, search results and the list beside a
video. New videos stay hidden until they are judged (a fraction of a second on a GPU, once the model is loaded), so
nothing flashes past. Each video is judged once and the answers are kept on your computer: hiding other kinds applies at
once, and a new rule asks each video once more.

## How accurate is it

Measured on 400 YouTube videos (the test half of the `youtube` suite of the
[benchmarks](../benchmarks.md#youtube-feed)), with the creator's own category as the label. The numbers are balanced
accuracy: the mean of the share of wanted videos kept and of unwanted ones hidden.

| model | your own rule (mean of 3) | hide music | hide gaming | only news | kinds: keep learning and news |
|---|---|---|---|---|---|
| **opendecider-nano** (this extension, on your computer) | **0.934** | **0.954** | 0.896 | 0.951 | 0.780 |
| TypeSafe Jev (in the cloud) | 0.940 | 0.931 | 0.919 | 0.971 | 0.865 |
| Laya, typed-decisions checkpoint | 0.870 | 0.901 | 0.759 | 0.949 | 0.740 |
| Laya | 0.852 | 0.916 | 0.714 | 0.927 | 0.712 |
| Laya, multilingual | 0.660 | 0.761 | 0.700 | 0.520 | 0.782 |

The rules, written as a viewer might: "Is this a music video or a song?", "Is this video about video games or
gameplay?", "Is this a news or current affairs video?".

- **Your own rule is where it is strongest**: within a point of Jev, TypeSafe's decision model in the cloud (ahead of
  it on music), and ahead of Laya, the other open decision model, on every rule.
- **The kinds are harder**, and there Jev leads by 8.5 points. Whether a cooking show is a how-to or entertainment is a
  judgement call, and the creator's category often disagrees with it (about one video in six, by our reading of
  Quietly's own keep and hide lists); opendecider-nano keeps more borderline videos than Jev does, so expect a few
  comedy and talk-show clips in the Focus feed.
- **Quietly's own request does not work with it.** Quietly asks Jev one question with the title inside the question
  and the same state every time; opendecider-nano judges the state, so it answers that request the same way for every
  video. OpenDecider Focus puts the video in the state instead.
- **English titles.** opendecider-nano was evaluated in English only, and so was this: titles in other languages and
  scripts were not measured.

## Speed and memory

| | with a GPU (WebGPU, fp16 build) | without one (WebAssembly, q8 build) |
|---|---|---|
| download, once | 755 MiB | 569 MiB |
| 40 videos, kinds | 1.1 s | 15 s |
| 40 videos, your rule | 0.3 s | 5.5 s |
| memory while loaded | about 3.3 GiB | about 3 GiB |

Times on an Apple M4 Max in Chrome (8 threads without the GPU); a slower computer takes longer.

The model is kept in the browser after the first download, and later loads in one or two seconds. It is loaded when
you open YouTube and closed after 10 minutes without filtering, which gives the memory back. Without a GPU, the feed
fills in a few videos at a time.

## Privacy and permissions

What leaves your computer: nothing you watch or search. The extension downloads the model's files from Hugging Face
once ([opendecider-nano-ONNX](https://huggingface.co/manjunathshiva/opendecider-nano-ONNX), at the revision
`@opendecider/web` pins), and checks each file's SHA-256 before using it. It makes no other network
request (its pages' Content Security Policy allows only those hosts), and it has no analytics.

| permission | why |
|---|---|
| read and change `www.youtube.com` | to read each video's title and channel, and hide the tiles you do not want |
| `storage` | your settings (synced by Chrome), and the answers already computed (on this computer) |
| `offscreen` | a hidden page that holds the model, where WebGPU and WebAssembly threads run |
| `contextMenus` | the right-click guard check |
| `alarms` | to close the model after 10 idle minutes |

The guard check reads only the text you select and right-click, and shows its answer in a small window.

## Build it yourself

```bash
(cd typescript && npm ci && npm run build)
(cd web && npm ci && npm run build)
cd extension && npm ci && npm run build     # then Load unpacked: extension/dist
npm test                                    # the logic
npx playwright install chromium && npm run e2e   # end to end, in a fresh Chromium profile
```

`npm run zip` builds the Web Store package; it rebuilds byte for byte from the same sources. After you rebuild, press
the reload button on `chrome://extensions`: Chrome keeps an unpacked extension's old service worker otherwise.

The idea of a feed filter that judges each title with a decision model comes from
[Quietly](https://github.com/joeydash/quietly) (MIT), which asks TypeSafe's Jev in the cloud. OpenDecider Focus is
written separately, and runs the model on your computer.
