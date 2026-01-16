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

# ==========================================
#        BUBBLE CLUSTER CONFIGURATION
# ==========================================
AUDIO_FILE = 'Perfect.mp3' 
OUTPUT_BASE = 'white_bubble_geometry' 

# --- Mode Selection ---
SHOW_PREVIEW = False 

# --- Visual Style (WHITE BACKGROUND) ---
NEON_COLOR_MAP = 'plasma'    # Vivid colors on white
BACKGROUND_COLOR = '#FFFFFF' # Pure White
GRID_COLOR = '#333333'       # Dark Grey grid
GRID_ALPHA = 0.15            # Subtle grid

# --- Bubble Physics (Mimicking the Original Image) ---
BUBBLE_COUNT = 30            # How many spheres make up the "blob"
BUBBLE_SPREAD = 0.15         # How tightly packed they are (Lower = denser blob)
MAX_BUBBLE_SIZE = 300        # Size of the largest sphere in the cluster

# --- Analysis Physics ---
N_MFCC = 20           
HOP_LENGTH = 512      
DURATION = 30         
K_NEIGHBORS = 6       
FPS = 60              

def get_unique_filename(base_name):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{base_name}_{timestamp}.mp4"

def generate_manifold_video(audio_path):
    print(f"\n--- INITIATING BUBBLE CLUSTER ENGINE ---")
    print(f"Target: {audio_path}")
    
    # 1. Load Audio
    try:
        y, sr = librosa.load(audio_path, duration=DURATION)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: File '{audio_path}' not found.")
        return

    # 2. Extract Features
    print(">> Extracting Psychoacoustic Features...")
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH).T
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]

    # Normalize
    scaler = StandardScaler()
    mfcc_scaled = scaler.fit_transform(mfcc)

    # 3. PCA Projection
    print(">> Projecting to 3D Manifold...")
    pca = PCA(n_components=3)
    mfcc_3d = pca.fit_transform(mfcc_scaled)

    # 4. Neural Web Connectivity
    print(f">> Weaving Web ({K_NEIGHBORS} connections)...")
    nbrs = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm='ball_tree').fit(mfcc_3d)
    _, indices = nbrs.kneighbors(mfcc_3d)

    # 5. Visual Setup
    print(">> Initializing Graphics Core...")
    fig = plt.figure(figsize=(16, 9), facecolor=BACKGROUND_COLOR)
    
    # --- FIT TO SCREEN ---
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax = fig.add_subplot(111, projection='3d', facecolor=BACKGROUND_COLOR)

    # --- ZOOM CAMERA ---
    ax.dist = 7  

    # --- CUSTOM AXES ---
    ax.xaxis.set_pane_color((1, 1, 1, 0))
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    
    ax.xaxis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.5, "alpha": GRID_ALPHA, "linestyle": "--"})
    ax.yaxis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.5, "alpha": GRID_ALPHA, "linestyle": "--"})
    ax.zaxis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.5, "alpha": GRID_ALPHA, "linestyle": "--"})

    ax.set_xlabel('TIMBRE X', color=GRID_COLOR, fontsize=8, labelpad=10)
    ax.set_ylabel('TIMBRE Y', color=GRID_COLOR, fontsize=8, labelpad=10)
    ax.set_zlabel('TIMBRE Z', color=GRID_COLOR, fontsize=8, labelpad=10)
    ax.tick_params(colors=GRID_COLOR, labelsize=7)
    
    ax.set_title(f'S O N I C   G E O M E T R Y   ::   {audio_path}', 
                 color='black', fontsize=14, fontname='Consolas', pad=10, alpha=0.8, y=0.98)

    # Color Mapping
    cmap = plt.get_cmap(NEON_COLOR_MAP)
    norm = plt.Normalize(vmin=np.min(centroid), vmax=np.max(centroid))
    all_colors = cmap(norm(centroid))
    
    # Pre-calculate normalized RMS for visual sizing
    rms_display = (rms / np.max(rms)) * 100
    rms_norm = rms / np.max(rms)

    # --- GRAPH OBJECTS ---
    # 1. History Trail (Standard dots)
    scat_history = ax.scatter([], [], [], alpha=0.5, edgecolors='none')
    
    # 2. The Head (BUBBLE CLUSTER)
    # We initialize with random data just to set the object type
    dummy_x = np.zeros(BUBBLE_COUNT)
    # 'o' marker for spheres, varying sizes will be applied in update
    scat_head = ax.scatter(dummy_x, dummy_x, dummy_x, c='red', alpha=0.9, edgecolors='none')
    
    # 3. Web Lines
    lines = [ax.plot([], [], [], color='black', alpha=0.15, linewidth=0.5)[0] for _ in range(K_NEIGHBORS)]
    
    # 4. Trajectory Line
    trajectory, = ax.plot([], [], [], color='#444444', alpha=0.6, linewidth=1.5)
    
    # 5. DATA HUD
    stats_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, 
                           color='black', fontsize=10, family='monospace', verticalalignment='top')

    # Set Limits
    ax.set_xlim(mfcc_3d[:,0].min(), mfcc_3d[:,0].max())
    ax.set_ylim(mfcc_3d[:,1].min(), mfcc_3d[:,1].max())
    ax.set_zlim(mfcc_3d[:,2].min(), mfcc_3d[:,2].max())

    # Pre-calculate random offsets for the bubble cluster so they jitter naturally
    # Shape: (Total Frames, Bubble Count, 3 Dimensions)
    cluster_noise = np.random.normal(0, BUBBLE_SPREAD, (len(mfcc_3d), BUBBLE_COUNT, 3))
    
    # Pre-calculate random sizes for the bubbles in the cluster
    # Some are big, some small, to look like a "Cluster"
    bubble_size_variance = np.random.uniform(0.2, 1.0, BUBBLE_COUNT)

    # Animation Params
    audio_frames_per_video_frame = (sr / HOP_LENGTH) / FPS
    total_video_frames = int(len(mfcc_3d) / audio_frames_per_video_frame)

    def update(frame_num):
        idx = int(frame_num * audio_frames_per_video_frame)
        if idx >= len(mfcc_3d): return scat_history,

        # Window logic for history
        tail_len = int(4 * (sr/HOP_LENGTH)) 
        start = max(0, idx - tail_len)
        
        # Current Position
        cx = mfcc_3d[idx, 0]
        cy = mfcc_3d[idx, 1]
        cz = mfcc_3d[idx, 2]
        
        # --- 1. UPDATE HISTORY TRAIL ---
        hist_x = mfcc_3d[start:idx, 0]
        hist_y = mfcc_3d[start:idx, 1]
        hist_z = mfcc_3d[start:idx, 2]
        
        scat_history._offsets3d = (hist_x, hist_y, hist_z)
        scat_history.set_color(all_colors[start:idx])
        # Simple size for history
        scat_history.set_sizes(np.full(len(hist_x), 20))
        
        # --- 2. UPDATE BUBBLE CLUSTER HEAD ---
        if idx < len(mfcc_3d):
            # Get pre-calculated noise for this frame
            offsets = cluster_noise[idx]
            
            # Position bubbles around the center point (cx, cy, cz)
            # We scale the spread by volume (Loud = Bigger cluster explosion)
            vol_spread = 1.0 + (rms_norm[idx] * 2.0)
            
            bx = cx + (offsets[:, 0] * vol_spread)
            by = cy + (offsets[:, 1] * vol_spread)
            bz = cz + (offsets[:, 2] * vol_spread)
            
            scat_head._offsets3d = (bx, by, bz)
            
            # Color: All bubbles match the note color
            scat_head.set_color(all_colors[idx])
            
            # Sizes: Base size * Random Variance * Volume Pulse
            base_size = MAX_BUBBLE_SIZE * rms_norm[idx]
            # Ensure at least some visibility
            base_size = max(base_size, 50) 
            
            final_sizes = base_size * bubble_size_variance
            scat_head.set_sizes(final_sizes)
            
            # --- UPDATE DATA HUD ---
            current_seconds = (idx * HOP_LENGTH) / sr
            mins = int(current_seconds // 60)
            secs = int(current_seconds % 60)
            ms = int((current_seconds * 100) % 100)
            
            hud_data = (
                f"TIME : {mins:02}:{secs:02}.{ms:02}\n"
                f"------------------\n"
                f"COORD X : {cx:+.2f}\n"
                f"COORD Y : {cy:+.2f}\n"
                f"COORD Z : {cz:+.2f}\n"
                f"------------------\n"
                f"DENSITY : {rms_display[idx]:03.0f}%\n"
                f"FREQ    : {centroid[idx]:.0f} Hz\n"
                f"FPS     : {FPS}"
            )
            stats_text.set_text(hud_data)
            stats_text.set_color('black') 

        # --- 3. UPDATE TRAJECTORY ---
        trajectory.set_data(hist_x, hist_y)
        trajectory.set_3d_properties(hist_z)

        # --- 4. UPDATE WEB ---
        current_neighbors = indices[idx]
        for i, neighbor_idx in enumerate(current_neighbors):
            lx = [cx, mfcc_3d[neighbor_idx, 0]]
            ly = [cy, mfcc_3d[neighbor_idx, 1]]
            lz = [cz, mfcc_3d[neighbor_idx, 2]]
            lines[i].set_data(lx, ly)
            lines[i].set_3d_properties(lz)
            lines[i].set_color(all_colors[idx]) 

        # Camera Rotation
        ax.view_init(elev=15, azim=frame_num * 0.3)
        return scat_history, scat_head, trajectory, stats_text, *lines

    # --- EXECUTION MODE ---
    ani = FuncAnimation(fig, update, frames=total_video_frames, blit=False, interval=1000/FPS)

    if SHOW_PREVIEW:
        print(">> STARTING LIVE PREVIEW...")
        plt.show()
    else:
        output_file = get_unique_filename(OUTPUT_BASE)
        print(f">> RENDERING to disk (Bubble Cluster Theme)...")
        print(f">> Processing {total_video_frames} frames at {FPS} FPS...")
        
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
        writer = FFMpegWriter(fps=FPS, metadata=dict(artist='CyberGeometry'), bitrate=8000)
        ani.save("temp_render.mp4", writer=writer, dpi=150)
        plt.close()

        # Audio Merge
        print(">> Merging High-Def Audio...")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.call([
            ffmpeg_exe, '-y',
            '-i', "temp_render.mp4",
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-shortest',
            output_file
        ])
        if os.path.exists("temp_render.mp4"): os.remove("temp_render.mp4")
        print(f"\nSUCCESS >> Saved to: {output_file}")

if __name__ == "__main__":
    generate_manifold_video(AUDIO_FILE)