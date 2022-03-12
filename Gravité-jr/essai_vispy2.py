import numpy as np
from vispy import app, scene


# Create canvas and view
canvas = scene.SceneCanvas(keys='interactive', size=(1200, 1200), show=True)
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera()
view.camera.scale_factor = 500

# Prepare data
np.random.seed(57983)
k = np.arange(-1, 1, 0.001)
liste_donnees = []
for i in k:
    liste_donnees.append([10 * np.cos(10 * np.pi * i) * np.sqrt(1-i**2), 10 * np.sin(10 * np.pi * i) * np.sqrt(1-i**2), 10 * i])
donnees = np.array(liste_donnees)

data = np.random.normal(size=(40, 3), loc=0, scale=100)
size = np.random.rand(40) * 100
colors = np.random.rand(40, 3)

data = np.concatenate([data, [[0, 0, 0]]], axis=0)
size = np.concatenate([size, [100]], axis=0)
colors = np.concatenate([colors, [[1, 0, 0]]], axis=0)


# Create and show visual
# vis = scene.visuals.Markers(
#     pos=donnees,
#     size=1,
#     antialias=0,
#     face_color=colors,
#     edge_color='white',
#     edge_width=0,
#     scaling=True,
#     spherical=True,
# )
# vis.parent = view.scene

lines = np.array([[data[i], data[-1]]
                  for i in range(len(data) - 1)])
line_vis = []

courbe = scene.visuals.Tube(donnees)
courbe.parent = view.scene

# for line in lines:
#     vis2 = scene.visuals.Tube(line, radius=5)
#     vis2.parent = view.scene
#     line_vis.append(vis2)

if __name__ == "__main__":
    app.run()