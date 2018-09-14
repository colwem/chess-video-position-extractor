import cv2
import numpy as np
import cairosvg
from chess_helpers import board_arrays_to_svgs


# Simplify the interface to cv2.VideoCapture
class VideoContainer(object):

    def __init__(self, fn):
        self.fn = fn
        self._current_frame = 0
        self._initialize_capture(self.fn)

    def get_video_array(self, start_at=0, end_at=None, chunk_size=None, overlap=0):
        if not end_at:
            end_at = self.frame_count
        total_frames = end_at - start_at

        if not chunk_size:
            chunk_size = total_frames

        overlap_array = []
        n = (total_frames // chunk_size) + 1
        self.seek_to(start_at)
        for i in range(n):
            current_frame_start = self._current_frame
            length = min(total_frames - i * chunk_size, chunk_size)
            buf = np.empty((length, self.frame_height, self.frame_width), dtype=np.float32)

            overlap_error = overlap
            if i == 0:
                overlap_error = 0
            else:
                buf[:overlap] = overlap_array

            for j in range(length - overlap_error):
                r = self.read()
                buf[j + overlap_error] = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            overlap_array = buf[-overlap:]

            yield buf, current_frame_start

    def get_frame_at(self, frame_number):
        current_frame = self._current_frame
        self.seet_to(frame_number)
        img = self.read()
        self.seek_to(current_frame)
        return img

    def _reinitialize(self):
        self._cap.release()
        self._initialize_capture(self.fn)

    def _initialize_capture(self, fn):
        self._cap = cv2.VideoCapture(fn, 0)
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_rate = self._cap.get(cv2.CAP_PROP_FPS)
        self._current_frame = 0
        self.length = self.frame_count

    def seek_to(self, frame):
        if self._current_frame < frame:
            while self._current_frame < frame:
                self.read()
        else:
            self._reinitialize()
            for _ in range(frame - 1):
                self.read(0)

        return self

    def read(self):
        self._current_frame += 1
        return self._cap.read()[1]

    def stream(self, start_at=0):
        self.seek_to(start_at)
        for i in range(self.length):
            yield self.read()

# Simplify the cv2.imshow gui interface
class Viewer(object):

    def __init__(self, window_name="Default Window", loc=(0, 0)):
        self._window_name = window_name
        self._loc = loc

        cv2.namedWindow(window_name)
        self.set_loc(loc)

    def update_frame(self, frame, frame_rate=30):
        wait = int(1000/frame_rate)
        cv2.imshow(self._window_name, frame)
        cv2.waitKey(wait) & 0xff

    def set_loc(self, loc=(0,0)):
        self._loc = loc
        cv2.moveWindow(self._window_name, *loc)
    def next_frame(self):
        self._current_frame += 1
        self.update_frame(self._video[self._current_frame])

    def play(self, video, verbose=True, frame_rate=1000/30, loop=False):
        while True:
            for i, frame in enumerate(video):
                self.update_frame(frame, int(1000/frame_rate))
                if verbose:
                    print('frame: ', i)
            if not loop:
                break

# SVG rendering functions
def surface_to_npim(surface):
    """ Transforms a Cairo surface into a numpy array. """
    im = +np.frombuffer(surface.get_data(), np.uint8)
    H,W = surface.get_height(), surface.get_width()
    im.shape = (H,W, 4) # for RGBA
    return im[:,:,:3]


def svg_to_npim(svg_bytestring, dpi=10):
    """ Renders a svg bytestring as a RGB image in a numpy array """
    tree = cairosvg.parser.Tree(bytestring=svg_bytestring)
    surf = cairosvg.surface.PNGSurface(tree,
                                       None,
                                       dpi,
                                       parent_width=250,
                                       parent_height=250).cairo
    return surface_to_npim(surf)


def render_svgs(svgs, skip=5, height=250, width=250):
    rendered = np.zeros((height, width))
    for i in range(len(svgs)):
        if not (i % skip):
            rendered = svg_to_npim(svgs[i])
        yield rendered


# Video stitching
def overlay_video(background, foreground, loc):
    return (overlay(bg, fg, loc)
            for bg, fg
            in zip(background, foreground))


def print_stream(file_name, source, view_while_writing=False):
    viewer = Viewer()
    frame = next(source)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = 828601953
    frame_rate = 29.96996465991079
    dim = frame.shape[1], frame.shape[0]
    out = cv2.VideoWriter(file_name, fourcc, frame_rate, dim)

    def do(frame):
        if view_while_writing:
            viewer.update_frame(frame, frame_rate=1000)
        out.write(frame)

    do(frame)
    for frame in source:
        do(frame)

    out.release()

def overlay(background, foreground, loc):
    x, y = loc
    width = foreground.shape[0]
    height = foreground.shape[1]
    background[x:x+width, y:y+height] = foreground
    return background



# Convenience functions
def show_together(file_name, board_arrays, start_at=0):
    v = VideoContainer(file_name)
    v.seek_to(start_at)

    original_viewer = Viewer('original', (0,0))
    rendered_viewer = Viewer('original', (0,500))

    i = start_at
    rendered = np.zeros((400,400))
    for svg in board_arrays_to_svgs(board_arrays)[start_at:]:
        if not (i % 4):
            rendered = svg_to_npim(svg)
        original = v.read()

        original_viewer.update_frame(original)
        rendered_viewer.update_frame(rendered)

        i += 1
        print(i)



def board_arrays_to_mp4(board_arrays, file_name="file.avi"):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # fourcc = 828601953
    frame_rate = 29.96996465991079
    out = cv2.VideoWriter(file_name, fourcc, frame_rate, (400, 400))
    a = np.zeros((400,400))
    i = 0
    for svg in board_arrays_to_svgs(board_arrays):
        if not (i % 2):
            a = svg_to_npim(svg)
        out.write(a)
        i += 1
        print(i)

    out.release()


def tile(tiles, start_at=0, end_at=0,
         add_frame_num=True, padding=5, title_bar_height=30):

    container_width = sum([tile.shape[1] + padding for tile in tiles[0]]) - padding
    container_height = sum([row[0].shape[2] + padding for row in tiles]) - padding + title_bar_height
    full_length = tiles[0][0].shape[0]
    if not end_at:
        end_at = full_length

    tiled = np.zeros(
        (end_at - start_at,
         container_height,
         container_width), dtype='uint8') + 255

    y = title_bar_height
    for i in range(len(tiles)):
        x = 0
        h = tiles[i][0].shape[2]
        for j in range(len(tiles[i])):
            tile = tiles[i][j]
            w = tile.shape[1]
            tiled[:, y:y + h, x: x + w] = tile[start_at:end_at]
            x += w + padding
        y += h + padding

    if add_frame_num:
        for i in range(len(tiled)):
            apply_number(tiled[i], i + start_at, (10, title_bar_height - 3))

    return tiled



def apply_number(frame, num, loc=(0,0)):

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = loc
    fontScale = 1
    fontColor = (10, 10, 10)
    lineType = 2

    cv2.putText(frame, str(num),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    return frame
