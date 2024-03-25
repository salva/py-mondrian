#!/usr/bin/env python3

import os, sys
import cv2
import numpy as np
import queue
import argparse
from math import sqrt
import pathlib

class Div:

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.children = None

    def best_cut(self, img, sep_width=0, min_band_size=5, error_function=None, error_delta=False):
        patch = img[self.y0:self.y1, self.x0:self.x1]
        area = (self.x1 - self.x0) * (self.y1 - self.y0)

        best_axis = None
        best_cut = None
        best_err = None
        all_err = None

        for axis in (0, 1):

            axis_len = patch.shape[axis]

            sum1 = np.sum(patch, axis=axis)
            sum2 = np.sum(patch**2, axis=axis)

            size = sum1.shape[0]
            if all_err is None:
                all_err, _ = error_function(np.sum(sum1, axis=0), np.sum(sum2, axis=0),
                                            axis_len, size)

            if size > min_band_size * 2 + sep_width:
                for i in range(min_band_size, size - min_band_size - sep_width):
                    i1 = i + sep_width

                    s1a = np.sum(sum1[:i], axis=0)
                    s2a = np.sum(sum2[:i], axis=0)
                    erra, wa = error_function(s1a, s2a, axis_len, i)
                    s1b = np.sum(sum1[i1:], axis=0)
                    s2b = np.sum(sum2[i1:], axis=0)
                    errb, wb = error_function(s1b, s2b, axis_len, size - i1)

                    if wa is None:
                        err = erra + errb
                    else:
                        err = (wa * erra + wb * errb) / (wa + wb)
                    if best_axis is None or err < best_err:
                        best_axis = axis
                        best_cut = i
                        best_err = err

        if best_axis is None:
            return None

        next_args = (best_axis, best_cut, best_cut + sep_width)
        if error_delta:
            return (all_err - best_err, next_args)

        return (all_err, next_args)

    def split(self, best_axis, best_cut, best_cut1):
        if best_axis == 0:
            children = (Div(self.x0, self.y0, self.x0 + best_cut, self.y1), Div(self.x0 + best_cut1, self.y0, self.x1, self.y1))
        else:
            children = (Div(self.x0, self.y0, self.x1, self.y0 + best_cut), Div(self.x0, self.y0 + best_cut1, self.x1, self.y1))

        self.children = children
        self.best_axis = best_axis
        self.best_cut = best_cut

        return children

    def draw_into(self, img, source_img, sep_width=0):
        if self.children is None:
            color = np.mean(source_img[self.y0:self.y1, self.x0:self.x1], axis=(0, 1))
            if sep_width > 0:
                drawing_area = (self.x1 - self.x0) * (self.y1 - self.y0)
                color_area = (self.x1 - self.x0 + sep_width) * (self.y1 - self.y0 + sep_width)
                color = color * color_area / drawing_area
                color = np.clip(color, 0.0, 1.0)
            img[self.y0:self.y1, self.x0:self.x1] = color
        else:
            for child in self.children:
                child.draw_into(img, source_img)

    def __lt__(self, other):
        return 0

def draw_frame(root, source_img, sep_width=0):
    img = np.zeros_like(source_img)
    root.draw_into(img, source_img)
    return (img*256).astype(np.uint8)


def error_acu_se(sum1, sum2, h, w):
    area = h * w
    return (np.sum(sum2 - sum1**2 / area), None)

def error_mean_se(sum1, sum2, h, w):
    area = h * w
    return (np.sum(sum2 / area - (sum1 / area)**2), area)

def error_linear_se(sum1, sum2, h, w):
    area = h * w
    sa = sqrt(area)
    return (np.sum(sum2 - sum1**2 / area) / sa, sa)

def error_len_se(sum1, sum2, h, w):
    area = h * w
    len = h**2 + w**2
    return (np.sum(sum2 / area - (sum1 / area)**2) * len, len)


def error_function_by_name(name):
    if name == 'acu-se':
        return error_acu_se
    elif name == 'mean-se':
        return error_mean_se
    elif name == 'linear-se':
        return error_linear_se
    elif name == 'len-se':
        return error_len_se
    else:
        raise ValueError('Unknown error function: {}'.format(name))

def write_redo():
    # get path to this script
    script_path = os.path.realpath(__file__)
    redo_path = os.path.join(os.path.dirname(script_path), 'redo.py')
    python_exe = sys.executable
    # write redo script
    with open(redo_path, 'w') as f:

        f.write(f"""#!{python_exe}

import os
import sys

cmd = [{repr(python_exe)}, {repr(script_path)}, *{repr(sys.argv[1:])}, *sys.argv[1:]]
print("running " + ' '.join(cmd))
os.execv({repr(python_exe)}, cmd)
""")

    # make redo script executable
    os.chmod(redo_path, 0o755)

def main():
    write_redo()

    parser = argparse.ArgumentParser(description='Mondrian')
    parser.add_argument('--source-img-path', type=str, default='guevos.jpeg', help='Path to the source image')
    parser.add_argument('--min-band-size', type=int, default=5, help='Minimum band size')
    parser.add_argument('--error-function', type=str, default='mean-se', help='Error function to use')
    parser.add_argument('--error-delta', action='store_true', help='Use error delta')
    parser.add_argument('--exp-frames', type=float, default=1, help='pick frmes with exponential spacing')
    parser.add_argument('--output-dir', type=str, default='output', help='Output directory for frames')
    parser.add_argument('--clean', action='store_true', help='Clean output directory')
    parser.add_argument('--mosaic-dims', type=str, default=None, help='Path to the mosaic image')
    parser.add_argument('--mosaic-skip', type=int, default=0, help='Skip frames for mosaic')
    parser.add_argument('--mosaic-randomize', action='store_true', help='Randomize mosaic image')
    parser.add_argument('--sep-width', type=int, default=0, help='Width of separator')
    parser.add_argument('--compensate-sep', action='store_true', help='Compensate separator width')

    args = parser.parse_args()

    source_img_path = args.source_img_path
    min_band_size = args.min_band_size
    sep_width = args.sep_width
    compensate_sep = args.compensate_sep
    error_function = error_function_by_name(args.error_function)
    error_delta = args.error_delta
    exp_frames = args.exp_frames

    output_dir = pathlib.Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        # remove all png files from output directory
        for f in os.listdir(output_dir):
            if f.endswith('.png'):
                os.remove(os.path.join(output_dir, f))

    img = cv2.imread(source_img_path) / 256.0

    mosaic_dims = args.mosaic_dims
    if mosaic_dims is not None:
        mosaic_width, mosaic_height = map(int, mosaic_dims.split('x'))
        mosaic_img = np.zeros((mosaic_height * (img.shape[0] + sep_width) - sep_width ,
                               mosaic_width * (img.shape[1] + sep_width) - sep_width,
                               3), dtype=np.uint8)

        if args.mosaic_randomize:
            mosaic_order = np.random.permutation(mosaic_width * mosaic_height)
        else:
            mosaic_order = np.arange(mosaic_width * mosaic_height)
        mosaic_skip = args.mosaic_skip

    cuts_queue = queue.PriorityQueue()
    root = Div(0, 0, img.shape[1], img.shape[0])
    weight, next_args = root.best_cut(img, sep_width=sep_width, min_band_size=min_band_size, error_function=error_function, error_delta=error_delta)
    cuts_queue.put((-weight, root, next_args))

    divisions_counter = 0
    next_frame_counter = 1
    frame_ix = 0

    while not cuts_queue.empty():
        divisions_counter += 1
        if divisions_counter >= next_frame_counter:
            frame_img = draw_frame(root, img, sep_width if compensate_sep else 0)
            cv2.imwrite(str(output_dir / ('frame_{:05d}_{:d}.png'.format(frame_ix, divisions_counter))), frame_img)

            if mosaic_dims is not None:
                if mosaic_skip <= frame_ix < len(mosaic_order) + mosaic_skip:
                    mosaic_ix = mosaic_order[frame_ix - mosaic_skip]
                    j = mosaic_ix // mosaic_width
                    i = mosaic_ix % mosaic_width
                    mosaic_img[j * (img.shape[0] + sep_width):(j + 1) * (img.shape[0] + sep_width) - sep_width,
                               i * (img.shape[1] + sep_width):(i + 1) * (img.shape[1] + sep_width) - sep_width] = frame_img

                    if frame_ix - mosaic_skip + 1 == len(mosaic_order):
                        cv2.imwrite(str(output_dir / 'mosaic.png'), mosaic_img)
                        print("Mosaic saved")

            next_frame_counter = int(next_frame_counter * exp_frames + 1)
            frame_ix += 1

        _, node, args = cuts_queue.get()
        children = node.split(*args)
        for child in children:
            try:
                weight, next_args = child.best_cut(img, sep_width=sep_width, min_band_size=min_band_size, error_function=error_function, error_delta=error_delta)
            except TypeError: # can't unpack None!
                pass
            else:
                cuts_queue.put((-weight, child, next_args))


    if mosaic_dims is not None:
        if next_frame_counter <= mosaic_height * mosaic_width:
            print("Warning: not enough frames to fill mosaic")

if __name__ == '__main__':
    main()
