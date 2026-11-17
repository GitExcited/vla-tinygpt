"""LIBERO task instructions per suite (the 10 tasks of each suite), used to sort the dataset's
episodes into suites. Matching ignores case, spaces and a final period."""

SUITE_TASKS = {
    "libero_spatial": [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl from table center and place it on the plate",
        "pick up the black bowl on the cookie box and place it on the plate",
        "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
        "pick up the black bowl on the ramekin and place it on the plate",
        "pick up the black bowl next to the cookie box and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
        "pick up the black bowl next to the plate and place it on the plate",
        "pick up the black bowl on the wooden cabinet and place it on the plate",
    ],
    "libero_object": [
        "pick up the alphabet soup and place it in the basket",
        "pick up the bbq sauce and place it in the basket",
        "pick up the butter and place it in the basket",
        "pick up the chocolate pudding and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the ketchup and place it in the basket",
        "pick up the milk and place it in the basket",
        "pick up the orange juice and place it in the basket",
        "pick up the salad dressing and place it in the basket",
        "pick up the tomato sauce and place it in the basket",
    ],
    "libero_goal": [
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
        "open the top drawer and put the bowl inside",
        "put the bowl on top of the cabinet",
        "push the plate to the front of the stove",
        "put the cream cheese in the bowl",
        "turn on the stove",
        "put the bowl on the plate",
        "put the wine bottle on the rack",
    ],
}


def norm(s: str) -> str:
    return " ".join(str(s).lower().strip().rstrip(".").split())


TASK_TO_SUITE = {norm(t): s for s, ts in SUITE_TASKS.items() for t in ts}
