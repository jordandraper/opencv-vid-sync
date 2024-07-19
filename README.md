# OpenCV Video Sync

Use OpenCV to synchronize two videos by determining the needed delay. Enjoy fast performance with multi-threading, which divides video files into chunks for better processing. Enhance your video editing with this efficient solution!

![match_found](https://github.com/jordandraper/opencv-vid-sync/assets/6191881/80715b01-2255-4a56-af94-d6bd829e298c)


https://github.com/jordandraper/opencv-vid-sync/assets/6191881/8a271865-4652-44d9-bfb2-30196445c4aa


<!-- ## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo -->

## Project Development

### Research and Initial Approach

During the initial phase, several existing solutions were explored, such as [align-videos-by-sound](https://github.com/jeorgen/align-videos-by-sound), which uses audio as a track to sync, along with other projects like [Sushi](https://github.com/tp7/Sushi), [ffsubsync](https://github.com/smacke/ffsubsync), and [AudioAlign](https://github.com/protyposis/AudioAlign).

The primary strategy for syncing two videos consisted of:
- Extracting individual frames as images using [ffmpeg](https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg)
- Selecting a frame from the first video and finding its match in the second video
- Comparing frames using the Structural Similarity Index Measure (SSIM)

### Improvements and Optimizations

Upon realizing that frame extraction was unnecessary, the following changes were implemented:
- Keeping frames in memory as objects, tracking frame number and timestamp
- Introducing threading to process video at multiple timestamps for faster average performance ([source](https://vuamitom.github.io/2019/12/13/fast-iterate-through-video-frames.html))
- Identifying scene transitions and matching them in both videos to reduce false positives

### Resources and Learning

To gain a deeper understanding of video analysis and tools, several resources were consulted:
- [Python: Compare Two Images](https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/)
- [Extract timestamps associated with frames ffmpeg extracts from a video](https://superuser.com/questions/841872/how-do-i-extract-the-timestamps-associated-with-frames-ffmpeg-extracts-from-a-vi)
- [Extract I-frames to images quickly](https://superuser.com/questions/1421133/extract-i-frames-to-images-quickly/1421195#1421195)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
- [Video Input with OpenCV and similarity measurement](https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html)
- [Structural similarity (Wikipedia)](https://en.wikipedia.org/wiki/Structural_similarity)

## Future Enhancements

- List all possible scene transitions
- Estimate match location in the first video and search around its timestamp in the second video, rather than brute-forcing through the entire video
- Implement speed optimizations and evaluate performance improvements, as suggested by [PyImageSearch](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)
- Incorporate audio sync, as mentioned in [this StackOverflow thread](https://stackoverflow.com/questions/25394937/automatically-sync-two-audio-recordings-in-python)
- Test alternative video frameworks for speed improvements, such as [Decord](https://github.com/dmlc/decord) and [VideoGear](https://towardsdatascience.com/lightning-fast-video-reading-in-python-c1438771c4e6)
- [x] Handle mismatched aspect ratios and potential cropping
- Prioritize search based on winning frame from consecutive SSIM in target SSIM
- Begin brute-force search within a window of the original timestamp, then expand on each failure while avoiding previously searched regions

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
