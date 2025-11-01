import psycopg2
from contextlib import contextmanager

class DatabaseConnection:
    def __init__(self, user, password, host, port, database):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.connection = None
        self.cursor = None

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            print("Connected to the database.")
        except Exception as error:
            print("Failed to connect to the database:", error)
            self.connection = None
            self.cursor = None

    def reconnect(self):
        if self.connection is not None:
            self.connection.close()
        self.connect()

    def execute(self, query, params=None):
        try:
            self.cursor.execute(query, params)
            self.connection.commit()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as error:
            print("Connection lost. Reconnecting...")
            self.reconnect()
            self.cursor.execute(query, params)
            self.connection.commit()
        except Exception as error:
            self.connection.rollback()
            raise error

    def fetchall(self):
        return self.cursor.fetchall()

    def fetchone(self):
        return self.cursor.fetchone()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

@contextmanager
def get_db_connection(db_connection):
    try:
        yield db_connection
    finally:
        db_connection.close()

def init_database_connection():
    db_connection = DatabaseConnection(
        user="",   #TODO
        password="",
        host="",
        port="",
        database="unlearn"
    )
    db_connection.connect()
    return db_connection

def add_benign_client_log(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'good client acc', {acc}, {e});""")
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'good client asr', {asr}, {e});""")
        
def add_l2_log_defender(l2, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'l2', {l2}, {e});""")

def add_attacker_client_log_final(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker client acc', {acc}, {e});""")
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker client asr', {asr}, {e});""")

def add_attacker_client_log_grad_simulate_train(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker simulate train acc', {acc}, {e});""")
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker grad simulate train asr', {asr}, {e});""")

def add_attacker_client_log_simulate_train(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker simulate train acc', {acc}, {e});""")
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker simulate train asr', {asr}, {e});""")

def add_attacker_client_log_gamma(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'attacker gamma asr', {asr}, {e});""")

def add_attacker_client_log_defender_before(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'defender before acc', {acc}, {e});""")
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, {client_idx}, 'defender before asr', {asr}, {e});""")
def add_global_log(acc, asr, r, client_idx, e, experiment_id, db_connection):
    #with get_db_connection(db_connection) as db:
        db = db_connection
        db.execute(f""" INSERT INTO DATA 
        (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
        ({experiment_id}, {r}, NULL, 'global acc', {acc}, {e});""")
        if asr is not None:
            db.execute(f""" INSERT INTO DATA 
            (experiment_id, ROUND, CLIENT_IDX, DATA_TYPE, VALUE, DATA_IDX) VALUES
            ({experiment_id}, {r}, {client_idx}, 'global asr', {asr}, {e});""")
if __name__ == "__mian__":
    with get_db_connection() as cursor:
        cursor.execute("""CREATE OR REPLACE FUNCTION delete_experiment_and_data(exp_name TEXT)
RETURNS VOID AS $$
BEGIN
    DELETE FROM data
    WHERE experiment_id IN (
        SELECT id FROM experiment WHERE name = exp_name
    );

    DELETE FROM experiment
    WHERE name = exp_name;
END;
$$ LANGUAGE plpgsql;
""")
    with get_db_connection() as cursor:
        cursor.execute("""CREATE OR REPLACE FUNCTION get_last_experiment_id()
RETURNS integer AS $$
DECLARE
    last_id INTEGER;
BEGIN
    SELECT ID INTO last_id FROM experiment
    ORDER BY starttime DESC
    LIMIT 1;
    RETURN LAST_ID;
END;
$$ LANGUAGE plpgsql;
""")
    with get_db_connection() as cursor:
        cursor.execute("""CREATE TABLE unlearn (
        UNLEARN_ID SERIAL NOT NULL PRIMARY KEY,
        experiment_ID integer REFERENCES experiment (ID),
        unlearn_type TEXT,
        asr REAL,
        acc REAL
        );""")
    with get_db_connection() as cursor:
        cursor.execute("""CREATE TABLE data (
        DATA_ID serial NOT NULL PRIMARY KEY,
        experiment_ID integer REFERENCES experiment (ID),
        ROUND integer,
        CLIENT_IDX integer,
        DATA_TYPE text,
        VALUE real,
        DATA_IDX integer
        );""")
    
    with get_db_connection() as cursor:
        cursor.execute("""CREATE TABLE experiment (
        ID serial NOT NULL PRIMARY KEY,
        name text,
        dataset text,
        starttime timestamp,
        FORGOT_CLIENT integer,
        SIMULATE_TRAIN BOOLEAN,
        GRAD_SIMULATE_TRAIN BOOLEAN,
        NOISE_THREAD integer,
        NUM_CLIENTS integer,
        COMMUNICATION_ROUNDS integer,
        PARTICIPANT_RATE integer,
        target_label integer,
        BATCH_SIZE integer,
        WARM_UP integer,
        local_epoch integer,
        ATTACKERS integer[],
        DEFENDERS integer[]
                       );""")

