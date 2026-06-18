def primary_irae_records(records):
    earliest_by_episode = {}
    irae_records = []

    for record in records:
        if record.get("condition_type") != "irae":
            continue

        time_to_onset = record.get("time_to_onset_months")
        time_start = record.get("time_start")
        if time_to_onset is None or time_start is None:
            continue

        episode_start = round(float(time_start) - float(time_to_onset), 2)
        key = (
            record.get("patient_id"),
            episode_start,
            record.get("associated_ici"),
            record.get("associated_treatment"),
        )
        onset = float(time_to_onset)
        earliest_by_episode[key] = min(onset, earliest_by_episode.get(key, onset))
        irae_records.append((key, onset, record))

    return [
        record
        for key, onset, record in irae_records
        if onset == earliest_by_episode[key]
    ]


def filter_primary_iraes(records, enabled):
    if not enabled:
        return list(records)

    primary_ids = {id(record) for record in primary_irae_records(records)}
    return [
        record
        for record in records
        if record.get("condition_type") != "irae" or id(record) in primary_ids
    ]
