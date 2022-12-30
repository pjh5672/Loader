import argparse
from pathlib import Path

import cv2

from loader import LoadImages
from utils import increment_path, imwrite

ROOT = Path(__file__).resolve().parents[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, help="Name to experiment")
    parser.add_argument("--path", type=str, required=True, help="Path to source")
    parser.add_argument("--img-size", type=int, default=320, help="Source size")
    parser.add_argument("--view", action="store_true", help="View mode for source")
    parser.add_argument("--save", action="store_true", help="Play with/without writing source")

    args = parser.parse_args()
    args.exp_path = ROOT / "experiment" / args.exp
    return args


def main(args, exist_ok=True, mkdir=True):
    save_dir = increment_path(path=args.exp_path, exist_ok=exist_ok, mkdir=mkdir)  # increment run
    vid_path, vid_writer = [None], [None]

    for path, im, im0, vid_cap, s in dataset:
        p = Path(path)
        save_path = str(save_dir / p.name)  # full path to source

        if args.view:
            cv2.imshow(str(p), im0)
            if dataset.mode == "image":
                key = cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                key = cv2.waitKey(10)  # 1 millisecond
                if key == ord('p'):
                    cv2.waitKey(-1) #wait until any key is pressed
                if key == 27:
                    cv2.destroyAllWindows()
                    break

        if args.save:
            if dataset.mode == "image":
                imwrite(save_path, im0)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)


if __name__ == "__main__":
    args = parse_args()
    dataset = LoadImages(path=args.path, img_size=args.img_size, transforms=None)
    main(args, exist_ok=False, mkdir=True)