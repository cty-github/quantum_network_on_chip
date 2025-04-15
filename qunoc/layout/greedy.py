from copy import deepcopy

import numpy as np

from .env import Package
from ..routing.router import DesignOrientation, DesignTemplate

max_score = -np.inf
package = Package(package_size=(50, 50))
for _ in range(100):
    if len(package.compute_modules) == 0:
        break
    area = package.compute_modules[0]
    package.compute_modules.pop(0)
    best_package = None
    for template in DesignTemplate:
        for orient in DesignOrientation:
            for interval in range(
                2 * np.max(package.fabricate_model.max_capacity),
                np.max(package.package_size),
                2
            ):
                try:
                    temp = deepcopy(package)
                    new_compute_areas, new_transfer_lines = temp.design_module_template(
                        area, template, interval, orient
                    )
                    temp.compute_modules.extend(new_compute_areas)
                    temp.transfer_lines.extend(new_transfer_lines)
                    temp.complete_package()
                    score = temp.estimate_transfer()[0]
                    if score > max_score:
                        max_score = score
                        best_package = temp
                except AssertionError:
                    pass
    if best_package is not None:
        package = best_package

print(max_score)
