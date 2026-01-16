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
#        LIGHT THEME CONFIGURATION
# ==========================================
AUDIO_FILE = 'song.mp3' 
OUTPUT_BASE = 'white_geometry_maximized' 

# --- Mode Selection ---
SHOW_PREVIEW = False 

# --- Visual Style (WHITE BACKGROUND) ---
NEON_COLOR_MAP = 'plasma'    # 'plasma' looks best on white (High Contrast)
BACKGROUND_COLOR = '#FFFFFF' # Pure White
GRID_COLOR = '#333333'       # Dark Grey grid
GRID_ALPHA = 0.15            # Subtle grid
NODE_SIZE_SCALE = 200        

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
    print(f"\n--- INITIATING LIGHT GEOMETRY ENGINE ---")
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
    
    # --- FIT TO SCREEN ADJUSTMENT ---
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    ax = fig.add_subplot(111, projection='3d', facecolor=BACKGROUND_COLOR)

    # --- ZOOM CAMERA ---
    ax.dist = 7  

    # --- CUSTOM AXES (Dark Lines for White Background) ---
    ax.xaxis.set_pane_color((1, 1, 1, 0)) # Transparent pane
    ax.yaxis.set_pane_color((1, 1, 1, 0))
    ax.zaxis.set_pane_color((1, 1, 1, 0))
    
    # Custom Grid Lines (Dark Grey Dashed)
    ax.xaxis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.5, "alpha": GRID_ALPHA, "linestyle": "--"})
    ax.yaxis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.5, "alpha": GRID_ALPHA, "linestyle": "--"})
    ax.zaxis._axinfo["grid"].update({"color": GRID_COLOR, "linewidth": 0.5, "alpha": GRID_ALPHA, "linestyle": "--"})

    # Axis Labels (Black Text)
    ax.set_xlabel('TIMBRE X', color=GRID_COLOR, fontsize=8, labelpad=10)
    ax.set_ylabel('TIMBRE Y', color=GRID_COLOR, fontsize=8, labelpad=10)
    ax.set_zlabel('TIMBRE Z', color=GRID_COLOR, fontsize=8, labelpad=10)
    ax.tick_params(colors=GRID_COLOR, labelsize=7)
    
    # Title (Black Text)
    ax.set_title(f'S O N I C   G E O M E T R Y   ::   {audio_path}', 
                 color='black', fontsize=14, fontname='Consolas', pad=10, alpha=0.8, y=0.98)

    # Color Mapping
    cmap = plt.get_cmap(NEON_COLOR_MAP)
    norm = plt.Normalize(vmin=np.min(centroid), vmax=np.max(centroid))
    all_colors = cmap(norm(centroid))
    all_sizes = (rms / np.max(rms)) * NODE_SIZE_SCALE
    
    rms_display = (rms / np.max(rms)) * 100

    # --- GRAPH OBJECTS ---
    # 1. History Trail (Darker transparency for visibility)
    scat_history = ax.scatter([], [], [], alpha=0.5, edgecolors='none')
    
    # 2. The Head (Dark outline to pop against white)
    scat_head = ax.scatter([], [], [], c='white', s=300, alpha=1.0, edgecolors='black', linewidth=1.5)
    
    # 3. Web Lines (Darker for visibility)
    lines = [ax.plot([], [], [], color='black', alpha=0.15, linewidth=0.5)[0] for _ in range(K_NEIGHBORS)]
    
    # 4. Trajectory Line (Black/Grey)
    trajectory, = ax.plot([], [], [], color='#444444', alpha=0.6, linewidth=1.5)
    
    # 5. DATA HUD (Black Text)
    stats_text = ax.text2D(0.05, 0.85, "", transform=ax.transAxes, 
                           color='black', fontsize=10, family='monospace', verticalalignment='top')

    # Set Limits
    ax.set_xlim(mfcc_3d[:,0].min(), mfcc_3d[:,0].max())
    ax.set_ylim(mfcc_3d[:,1].min(), mfcc_3d[:,1].max())
    ax.set_zlim(mfcc_3d[:,2].min(), mfcc_3d[:,2].max())

    # Animation Params
    audio_frames_per_video_frame = (sr / HOP_LENGTH) / FPS
    total_video_frames = int(len(mfcc_3d) / audio_frames_per_video_frame)

    def update(frame_num):
        idx = int(frame_num * audio_frames_per_video_frame)
        if idx >= len(mfcc_3d): return scat_history,

        # Window logic
        tail_len = int(4 * (sr/HOP_LENGTH)) 
        start = max(0, idx - tail_len)
        
        # Coordinates
        cx = mfcc_3d[start:idx, 0]
        cy = mfcc_3d[start:idx, 1]
        cz = mfcc_3d[start:idx, 2]
        
        # Update History
        scat_history._offsets3d = (cx, cy, cz)
        scat_history.set_color(all_colors[start:idx])
        scat_history.set_sizes(all_sizes[start:idx])
        
        # Update Head
        if idx < len(mfcc_3d):
            current_x = mfcc_3d[idx, 0]
            current_y = mfcc_3d[idx, 1]
            current_z = mfcc_3d[idx, 2]
            
            scat_head._offsets3d = ([current_x], [current_y], [current_z])
            scat_head.set_color(all_colors[idx])
            # Pulsing effect
            pulse = all_sizes[idx] * 2
            scat_head.set_sizes([pulse])
            
            # --- UPDATE DATA HUD ---
            current_seconds = (idx * HOP_LENGTH) / sr
            mins = int(current_seconds // 60)
            secs = int(current_seconds % 60)
            ms = int((current_seconds * 100) % 100)
            
            hud_data = (
                f"TIME : {mins:02}:{secs:02}.{ms:02}\n"
                f"------------------\n"
                f"COORD X : {current_x:+.2f}\n"
                f"COORD Y : {current_y:+.2f}\n"
                f"COORD Z : {current_z:+.2f}\n"
                f"------------------\n"
                f"DENSITY : {rms_display[idx]:03.0f}%\n"
                f"FREQ    : {centroid[idx]:.0f} Hz\n"
                f"FPS     : {FPS}"
            )
            stats_text.set_text(hud_data)
            # Tint text slightly for style (darker version of note color)
            stats_text.set_color('black') 

        # Update Trajectory
        trajectory.set_data(cx, cy)
        trajectory.set_3d_properties(cz)

        # Update Web
        current_neighbors = indices[idx]
        for i, neighbor_idx in enumerate(current_neighbors):
            lx = [mfcc_3d[idx, 0], mfcc_3d[neighbor_idx, 0]]
            ly = [mfcc_3d[idx, 1], mfcc_3d[neighbor_idx, 1]]
            lz = [mfcc_3d[idx, 2], mfcc_3d[neighbor_idx, 2]]
            lines[i].set_data(lx, ly)
            lines[i].set_3d_properties(lz)
            lines[i].set_color(all_colors[idx]) 

        # Camera Rotation
        ax.view_init(elev=15, azim=frame_num * 0.3)
        return scat_history, scat_head, trajectory, stats_text, *lines

    # --- EXECUTION MODE ---
    ani = FuncAnimation(fig, update, frames=total_video_frames, blit=False, interval=1000/FPS)

    if SHOW_PREVIEW:
        print(">> STARTING LIVE PREVIEW (Close window to stop)...")
        plt.show()
    else:
        output_file = get_unique_filename(OUTPUT_BASE)
        print(f">> RENDERING to disk (White Theme)...")
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