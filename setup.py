from setuptools import find_packages, setup 
from typing import List 

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
  """
    The function will return the list of packages.
  """
  requirements = []
  
  with open(file_path, 'r') as file_obj:
    requirements = file_obj.readlines()
    requirements = [req.replace("\n", "").strip() for req in requirements]
    
    if HYPHEN_E_DOT in requirements:
      requirements.remove(HYPHEN_E_DOT)
      
  return requirements

setup(
  name='The Autonomous Customer Support Agent Copilot for Orange Burkina',
  description="A fully RAG app built to reduce the customers' waiting time when calling an Orange call center for help. Hence creating high business value.",
  version="0.0.1",
  author="Abddoul-Rahmane BANDE",
  author_email="abdoulrahmanebande@gmail.com",
  packages=find_packages(),
  install_requires=get_requirements('requirements.txt')
)