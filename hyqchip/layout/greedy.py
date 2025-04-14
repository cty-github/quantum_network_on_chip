from copy import deepcopy

import numpy as np

from hyqchip.layout.env import Package
from hyqchip.routing.router import design_orientations, design_templates

max_score = -np.inf
package = Package(package_size=(50, 50))
for _ in range(100):
    if len(package.compute_areas) == 0:
        break
    area = package.compute_areas[0]
    package.compute_areas.pop(0)
    best_package = None
    for template in design_templates:
        for orient in design_orientations:
            for interval in range(
                2 * np.max(package.fabricate_model.max_die_size),
                np.max(package.package_size),
                2
            ):
                try:
                    temp = deepcopy(package)
                    new_compute_areas, new_transfer_lines = temp.design_area(
                        area, template, interval, orient
                    )
                    temp.compute_areas.extend(new_compute_areas)
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
