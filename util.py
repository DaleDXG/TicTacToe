
def add_coordinates(coordinates_1, coordinates_2):
    # could check DimentionMatchingError
    result = []
    for i in range(len(coordinates_1)):
        result.append(coordinates_1[i] + coordinates_2[i])
    return tuple(result)

def multiply_coordinates(coordinates, multiplier):
    result = []
    for i in range(len(coordinates)):
        result.append(coordinates[i] * multiplier)
    return tuple(result)

def copy_list(list):
    result = []
    for item in list:
        result.append(item)
    return result