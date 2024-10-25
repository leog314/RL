Requirements:  
  Python (<=3.11)
  to install all needed libraries run:

    pip install torch torchvision gymnasium matplotlib numpy
    
  Also you'll need some gymnasium-environments to test the algorithm on, for example the Box2d-env (https://gymnasium.farama.org/environments/box2d/).
  Because of this, you propably need to install additional packages like (for Box2d):
  
    pip install swig
    pip install gymnasium[box2d]

  Needed packages are listed at the specific websites of the environments (see also: https://gymnasium.farama.org/)
