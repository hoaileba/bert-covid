DROP TABLE IF EXISTS rest_dialogue;
DROP TABLE IF EXISTS rest_request_dia;
DROP TABLE IF EXISTS rest_ans;



CREATE TABLE rest_dialogue (
	id_dia	integer NOT NULL,
	PRIMARY KEY(id_dia)
);
CREATE TABLE rest_request_dia (
	id_re	integer NOT NULL,
	req 	text NOT NULL,
	hero	text,
	skill	text,
	id_dia_id integer NOT NULL,
	intent	text,
	action	text,
	FOREIGN KEY(id_dia_id) REFERENCES rest_dialogue(id_dia) DEFERRABLE INITIALLY DEFERRED,
	PRIMARY KEY(id_re)

CREATE TABLE rest_ans (
	id	integer NOT NULL PRIMARY KEY AUTOINCREMENT,
	champ	text NOT NULL,
	introduce	text,
	how_to_play	text,
	how_to_use_skill_Q	text,
	how_to_use_skill_W	text,
	how_to_use_skill_E	text,
    how_to_use_skill_R	text,
	skill_up	text,
	build_item	text,
	support_socket	text,
	counter	text,
	be_countered	text,
	combo	text,
	combine_with	text
);