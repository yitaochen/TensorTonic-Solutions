def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    # Write code here
    errors = []
    for record_index, record in enumerate(records):
        is_valid = True 
        error = []
        for col_schema in schema:
            col_name = col_schema['column']
            if col_name not in record:
                is_valid = False
                error.append(f"{col_name}: missing")
                continue
            col_nullable = col_schema['nullable']
            if not col_nullable and record[col_name] is None:
                is_valid = False
                error.append(f"{col_name}: null")
                continue
            elif col_nullable and record[col_name] is None:
                continue
            col_type = col_schema['type']
            actual_type = type(record[col_name]).__name__
            if col_type == "float" and actual_type in ["float", "int"]:
                actual_type = "float"
            if actual_type != col_type:
                is_valid = False
                error.append(f"{col_name}: expected {col_type}, got {actual_type}")
                continue 
            record_val = record[col_name]
            if "min" in col_schema and "max" in col_schema:
                col_min = col_schema['min']
                col_max = col_schema['max']
                if not col_min <= record_val <= col_max:
                    is_valid = False 
                    error.append(f"{col_name}: out of range")
        errors.append((record_index, is_valid, error))
            

    return errors 