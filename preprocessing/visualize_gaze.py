import cv2
import pandas as pd
import numpy as np
import os
import argparse

# -----------------------------
# Command-line interface
# -----------------------------
parser = argparse.ArgumentParser(description="Overlay gaze on videos.")
parser.add_argument('--csv', required=True, help='Path to CSV file')
parser.add_argument('--videos', nargs='+', required=True, help='List of video files')
parser.add_argument('--display_width', type=int, required=True, help='Display width for gaze coordinates')
parser.add_argument('--display_height', type=int, required=True, help='Display height for gaze coordinates')
parser.add_argument('--output_width', type=int, default=1280, help='Output video width')
parser.add_argument('--output_height', type=int, default=720, help='Output video height')
parser.add_argument('--trail_length', type=int, default=5, help='Number of frames to show in gaze trail')
args = parser.parse_args()

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(args.csv)

# -----------------------------
# Parameters
# -----------------------------
trail_colors = [(200,200,255), (150,150,255), (100,100,255), (50,50,255)]
current_color = (0,0,255)

# -----------------------------
# Process each video
# -----------------------------
for video_file in args.videos:
    print(f"Processing {video_file}...")

    # Safe exact match for video names after '|'
    if 'events' not in df.columns:
        raise ValueError("CSV must contain an 'events' column.")

    event_videos = df['events'].astype(str).str.split('|').str[-1]
    video_events = df[event_videos == os.path.basename(video_file)]

    if len(video_events) < 2:
        print(f"Warning: Could not find start/stop for {video_file}. Skipping.")
        continue

    start_idx = video_events.index[0]
    end_idx = video_events.index[-1]
    gaze_df = df.loc[start_idx:end_idx].copy()

    # Filter out-of-bounds gaze points
    valid_mask = (
        (gaze_df['gaze_x'] >= 0) & (gaze_df['gaze_x'] <= args.display_width) &
        (gaze_df['gaze_y'] >= 0) & (gaze_df['gaze_y'] <= args.display_height)
    )
    gaze_df = gaze_df[valid_mask]

    # Extract and rescale gaze
    trial_times = gaze_df['time'].to_numpy()
    #trial_times = gaze_df['trial_time'].to_numpy()
    gazeX = gaze_df['gaze_x'].to_numpy() * (args.output_width / args.display_width)
    gazeY = gaze_df['gaze_y'].to_numpy() * (args.output_height / args.display_height)

    # Open video
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare output video
    base_name = os.path.splitext(os.path.basename(video_file))[0]
    output_file = f"{base_name}_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (args.output_width, args.output_height))

    trail_history = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (args.output_width, args.output_height))

        # Select gaze for this frame
        t_start = frame_idx / fps
        t_end = (frame_idx + 1) / fps
        mask = (trial_times >= t_start) & (trial_times < t_end)

        if np.any(mask):
            # Compute NaN-safe means
            avg_x = np.nanmean(gazeX[mask])
            avg_y = np.nanmean(gazeY[mask])

            # If ALL values were NaN, nanmean returns NaN → skip this frame
            if np.isnan(avg_x) or np.isnan(avg_y):
                current_point = None
            else:
                current_point = (int(avg_x), int(avg_y))
                trail_history.append(current_point)
        else:
            current_point = None

        # Trim trail history
        trail_history = trail_history[-args.trail_length:]

        # Draw trail (older = lighter)
        for i, point in enumerate(trail_history[:-1]):
            color_idx = min(len(trail_history) - i - 2, len(trail_colors) - 1)
            cv2.circle(frame, point, 8, trail_colors[color_idx], -1)

        # Draw current gaze point
        if current_point is not None:
            cv2.circle(frame, current_point, 10, current_color, -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Saved annotated video to {output_file}")

print("All videos processed.")
