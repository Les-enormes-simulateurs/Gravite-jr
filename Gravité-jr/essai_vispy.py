import sys

from conversion_vers_vispy import scene
from conversion_vers_vispy.visuals.transforms import STTransform

canvas = scene.SceneCanvas(keys='interactive', bgcolor='black',
                           size=(800, 600), show=True)

view = canvas.central_widget.add_view()
view.camera = 'arcball'

sphere1 = scene.visuals.Sphere(radius=1, method='latitude', parent=view.scene)

sphere2 = scene.visuals.Sphere(radius=1, method='ico', parent=view.scene)

sphere3 = scene.visuals.Sphere(radius=1, rows=10, cols=10, depth=10,
                               method='cube', parent=sphere2)

sphere1.transform = STTransform(translate=[-2.5, 0, 0])
sphere3.transform = STTransform(translate=[2.5, 0, 0])

view.camera.set_range(x=[-3, 3])

if __name__ == '__main__' and sys.flags.interactive == 0:
    canvas.app.run()
