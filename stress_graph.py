
import csv
from collections import deque
from datetime import datetime

import matplotlib.pyplot as plt


class StressGraph:
    def __init__(self, max_points=100):
        # Data shown on the live graph (limited to the latest 100 points)
        self.scores = deque(maxlen=max_points)
        self.frames = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)

        # Complete session history (used for CSV export)
        self.all_scores = []
        self.all_frames = []
        self.all_timestamps = []

        self.frame_count = 0

        # Configure the live graph
        plt.ion()
        self.fig, self.ax = plt.subplots()

        self.fig.canvas.manager.set_window_title(
            "Stress Score Graph"
        )

        # Position graph on the right side of the screen (Windows)
        try:
            self.fig.canvas.manager.window.wm_geometry("+850+50")
        except (AttributeError, TypeError):
            pass

        self.line, = self.ax.plot(
            [], [],
            label="Experimental score"
        )

        self.ax.set_title("Experimental Score History")
        self.ax.set_xlabel("Recent frames")
        self.ax.set_ylabel("Score (0–100)")
        self.ax.set_ylim(0, 100)
        self.ax.grid(True)
        self.ax.legend()

    def update(self, score):
        """Record one score and update the live graph."""
        self.frame_count += 1

        timestamp = datetime.now().isoformat(
            timespec="milliseconds"
        )

        score = float(score)

        # Update the live graph history
        self.frames.append(self.frame_count)
        self.scores.append(score)
        self.timestamps.append(timestamp)

        # Preserve the entire session for CSV export
        self.all_frames.append(self.frame_count)
        self.all_scores.append(score)
        self.all_timestamps.append(timestamp)

        # Update plotted data
        self.line.set_data(
            list(self.frames),
            list(self.scores)
        )

        if self.frames:
            left = self.frames[0]
            right = self.frames[-1]

            # Avoid identical left and right axis limits
            if left == right:
                right = left + 1

            self.ax.set_xlim(left, right)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def save_history(self, filename="stress_history.csv"):
        """Save every recorded score from this session to CSV."""
        with open(
            filename,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow([
                "Frame",
                "Experimental Score",
                "Timestamp"
            ])

            writer.writerows(
                zip(
                    self.all_frames,
                    self.all_scores,
                    self.all_timestamps
                )
            )

        print(
            f"Saved {len(self.all_scores)} readings to {filename}"
        )
    
    def print_session_summary(self):
        """Print summary statistics for the current session."""

        if not self.all_scores:
            print("No readings recorded in this session.")
            return

        average_score = sum(self.all_scores) / len(self.all_scores)
        maximum_score = max(self.all_scores)
        minimum_score = min(self.all_scores)

        max_index = self.all_scores.index(maximum_score)
        max_time = self.all_timestamps[max_index]

        start_time = datetime.fromisoformat(self.all_timestamps[0])
        end_time = datetime.fromisoformat(self.all_timestamps[-1])
        duration = end_time - start_time

        print("\n========== SESSION SUMMARY ==========")
        print(f"Total readings       : {len(self.all_scores)}")
        print(f"Average score        : {average_score:.2f}/100")
        print(f"Maximum score        : {maximum_score:.2f}/100")
        print(f"Minimum score        : {minimum_score:.2f}/100")
        print(f"Maximum score time   : {max_time}")
        print(f"Session duration     : {duration}")
        print("=====================================\n")

    def close(self):
        """Close the graph window."""
        plt.close(self.fig)
