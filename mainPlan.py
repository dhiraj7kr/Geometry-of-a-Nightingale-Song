import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.cm as cm
import subprocess
import os
import imageio_ffmpeg
import time

# --- Configuration ---
AUDIO_FILE = 'nightingale.mp3' 
# Base name for output (timestamp will be added automatically)
OUTPUT_BASE = 'nightingale_geometry' 

# Analysis Settings
N_MFCC = 20           # Higher = more complex texture analysis
HOP_LENGTH = 512      # Lower = smoother animation but slower rendering
DURATION = 30         # Seconds to analyze (None for full song)
K_NEIGHBORS = 8       # Increased neighbors for a richer "web"
FPS = 30              # Frames Per Second
DPI = 150             # Resolution (150 is good for HD, 300 for 4K)

def get_unique_filename(base_name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_name}_{timestamp}.mp4"

def generate_manifold_video(audio_path):
    output_file = get_unique_filename(OUTPUT_BASE)
    print(f"--- Processing: {audio_path} ---")
    print(f"--- Output will be: {output_file} ---")

    try:
        y, sr = librosa.load(audio_path, duration=DURATION)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{audio_path}'. Check the file name!")
        return

    # 1. Feature Extraction
    print("Extracting high-fidelity audio features...")
    # MFCCs (Timbre / Shape)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).T
    # Spectral Centroid (Color / Pitch)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    # RMS Energy (Size / Loudness)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    # Normalize data for PCA
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc)

    # 2. Dimensionality Reduction (PCA)
    print("Calculating 3D Manifold Coordinates...")
    pca = PCA(n_components=3)
    mfcc_3d = pca.fit_transform(mfcc_scaled)

    # 3. Geometric Connectivity (The "Web")
    print(f"Computing mesh connectivity ({K_NEIGHBORS} neighbors)...")
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree').fit(mfcc_3d)
    _, indices = nbrs.kneighbors(mfcc_3d)

    # 4. Visualization Setup (Scientific Style)
    print("Setting up High-Resolution 3D Canvas...")
    fig = plt.figure(figsize=(16, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # --- AXIS STYLING (The "Scientific" Look) ---
    # Make panes transparent but keep grid
    ax.xaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.yaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))
    ax.zaxis.set_pane_color((0.1, 0.1, 0.1, 1.0))

    # Color the grid lines and axes white
    ax.xaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0.2)
    ax.yaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0.2)
    ax.zaxis._axinfo["grid"]['color'] =  (1, 1, 1, 0.2)

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')

    ax.set_xlabel('PCA 1 (Timbre X)', color='white', fontsize=10)
    ax.set_ylabel('PCA 2 (Timbre Y)', color='white', fontsize=10)
    ax.set_zlabel('PCA 3 (Timbre Z)', color='white', fontsize=10)
    ax.set_title(f'Acoustic Geometry: {audio_path}', color='white', fontsize=14, pad=20)

    # Color Mapping
    cmap = cm.plasma # Plasma is very vibrant and high-contrast
    norm = plt.Normalize(vmin=np.min(centroid), vmax=np.max(centroid))
    all_colors = cmap(norm(centroid))
    # Scale sizes for better visibility
    all_sizes = (rms / np.max(rms)) * 150 

    # --- PLOT ELEMENTS ---
    # 1. The History Trail (Faint, showing the path)
    scat_history = ax.scatter([], [], [], c=[], s=[], alpha=0.3, edgecolors='none')
    
    # 2. The "Head" (Current Note - Bright and Large)
    scat_head = ax.scatter([], [], [], c='white', s=200, alpha=1.0, edgecolors='white')

    # 3. The "Web" Lines (Connectivity)
    # Created as a list of line objects
    lines = [ax.plot([], [], [], color='white', alpha=0.15, linewidth=0.4)[0] for _ in range(K_NEIGHBORS)]
    
    # 4. The Trajectory Line (Immediate path)
    trajectory, = ax.plot([], [], [], color='white', alpha=0.6, linewidth=1.5)

    # Set Camera Limits
    ax.set_xlim(mfcc_3d[:,0].min(), mfcc_3d[:,0].max())
    ax.set_ylim(mfcc_3d[:,1].min(), mfcc_3d[:,1].max())
    ax.set_zlim(mfcc_3d[:,2].min(), mfcc_3d[:,2].max())

    # Timing calculations
    audio_frames_per_video_frame = (sr / HOP_LENGTH) / FPS
    total_video_frames = int(len(mfcc_3d) / audio_frames_per_video_frame)

    print(f"Rendering {total_video_frames} frames at {FPS} FPS...")

    def update(frame_num):
        idx = int(frame_num * audio_frames_per_video_frame)
        if idx >= len(mfcc_3d): return scat_history,
        
        # Define window for history trail (last 3 seconds)
        tail_len = int(3 * (sr/HOP_LENGTH)) 
        start = max(0, idx - tail_len)
        
        # Get coordinates
        curr_x = mfcc_3d[start:idx, 0]
        curr_y = mfcc_3d[start:idx, 1]
        curr_z = mfcc_3d[start:idx, 2]
        
        # Update History Scatter
        scat_history._offsets3d = (curr_x, curr_y, curr_z)
        scat_history.set_color(all_colors[start:idx])
        scat_history.set_sizes(all_sizes[start:idx])
        
        # Update Head Scatter (The single leading point)
        if idx < len(mfcc_3d):
            scat_head._offsets3d = ([mfcc_3d[idx, 0]], [mfcc_3d[idx, 1]], [mfcc_3d[idx, 2]])
            # Make the head pulse with volume
            current_size = all_sizes[idx] * 2  
            scat_head.set_sizes([current_size])
            scat_head.set_color(all_colors[idx])

        # Update Trajectory Line
        trajectory.set_data(curr_x, curr_y)
        trajectory.set_3d_properties(curr_z)

        # Update Web Connections (Nearest Neighbors)
        current_neighbors = indices[idx]
        for i, neighbor_idx in enumerate(current_neighbors):
            lx = [mfcc_3d[idx, 0], mfcc_3d[neighbor_idx, 0]]
            ly = [mfcc_3d[idx, 1], mfcc_3d[neighbor_idx, 1]]
            lz = [mfcc_3d[idx, 2], mfcc_3d[neighbor_idx, 2]]
            lines[i].set_data(lx, ly)
            lines[i].set_3d_properties(lz)
            # Dynamic line color based on pitch
            lines[i].set_color(all_colors[idx])

        # Smooth Camera Rotation
        ax.view_init(elev=20, azim=frame_num * 0.15)
        
        return scat_history, scat_head, trajectory, *lines

    # 5. Render Video
    temp_filename = "temp_render.mp4"
    ani = FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    
    # Force FFmpeg path for Matplotlib
    plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    
    # High Bitrate for quality
    writer = FFMpegWriter(fps=FPS, metadata=dict(artist='AudioGeometry'), bitrate=8000)
    ani.save(temp_filename, writer=writer, dpi=DPI)
    plt.close()

    # 6. Merge Audio
    print("Merging Audio and Video...")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    
    subprocess.call([
        ffmpeg_exe, '-y',
        '-i', temp_filename,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        output_file
    ])

    if os.path.exists(temp_filename):
        os.remove(temp_filename)
        
    print(f"Success! High-quality render saved to: {output_file}")

if __name__ == "__main__":
    generate_manifold_video(AUDIO_FILE)