import subprocess as sp


class VideoWriter:
    class Error(Exception):
        pass

    def __init__(self, w, h, fps=25, output_file="video.mp4"):
        self.command = [
            'ffmpeg',
            '-y',                    # overwrite output file if it exists
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', '%dx%d' % (w, h),  # size of one frame
            '-pix_fmt', 'bgr24',
            '-r', str(fps),          # frames per second
            '-i', '-',               # The imput comes from a pipe
            '-an',                   # Tells FFMPEG not to expect any audio
            '-vcodec', 'mpeg4',
            output_file,
        ]
        self.ffmpeg = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)

    def add_frame(self, frame):
        try:
            self.ffmpeg.stdin.write(frame.tostring())
        except IOError:
            reason = self.ffmpeg.stderr.read()
            raise self.Error(reason)

    def close(self):
        self.ffmpeg.terminate()


# Usage example
if __name__ == "__main__":
    import numpy as np

    w, h = 640, 480
    video = VideoWriter(w, h)
    for i in range(100):
        video.add_frame(np.random.randint(0, 255, (h, w, 3)))
