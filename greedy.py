from copy import deepcopy

import numpy as np

from hyqchip.layout.env import Package
from hyqchip.routing.router import design_orientations, design_templates

package = Package()
for _ in range(1):
    area = package.compute_areas[0]
    package.compute_areas.pop(0)
    max_score = -np.inf
    best_package = None
    for template in design_templates:
        for orient in design_orientations:
            for interval in range(0, np.max(package.package_size)):
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
