# Project Title

One Paragraph of project description goes here

## Getting Started

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

End with an example of getting some data out of the system or using it for a little demo

## Deployment

Add additional notes about how to deploy this on a live system

## Project Development

First researching available solutions resulted in finding https://github.com/jeorgen/align-videos-by-sound
Similar goals but different implementations: https://github.com/tp7/Sushi, https://github.com/smacke/ffsubsync, https://github.com/protyposis/AudioAlign

The initial strategy to accomplish the solution of syncing two videos was as follows:

    - extract individual frames as images using ffmpeg and compare from those images
        - https://stackoverflow.com/questions/10957412/fastest-way-to-extract-frames-using-ffmpeg

Then realized frame extraction was unnecessary, as the images are represented as Numpy arrays and could be left in memory:
    - frames can just be kept in memory as objects and we just keep track of which frame we are dealing with
    - Introduce threading to process the video at multiple timestamps. The thought being that this would decrease the average time, and increase the worst case time. https://vuamitom.github.io/2019/12/13/fast-iterate-through-video-frames.html

I had very little knowledge of video analysis and tools. Here were some useful resources during my learning:
    - https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    - https://superuser.com/questions/841872/how-do-i-extract-the-timestamps-associated-with-frames-ffmpeg-extracts-from-a-vi
    - https://superuser.com/questions/1421133/extract-i-frames-to-images-quickly/1421195#1421195
    - https://github.com/kkroening/ffmpeg-python
    - https://docs.opencv.org/2.4/doc/tutorials/highgui/video-input-psnr-ssim/video-input-psnr-ssim.html
    - https://en.wikipedia.org/wiki/Structural_similarity

## Still to do

    - expand codebase to list all possible scene transitions
    - guess match location and search around it instead of brute forcing through entire video
    - implement speed fixes https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
    - add in sync via audio: https://stackoverflow.com/questions/25394937/automatically-sync-two-audio-recordings-in-python
    - test other video frameworks: 
        - https://github.com/dmlc/decord
        - https://towardsdatascience.com/lightning-fast-video-reading-in-python-c1438771c4e6
    - [x] handle mismatched aspect ratios and potention cropping 
    - target search based on winning frame from consecutive ssim to priority search in taget ssim
    
# start brute force search within a window of original timestamp, then expand on each failure and avoid previously searched region

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

