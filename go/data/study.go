// Copyright 2024 The Zimtohrli Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package data contains structs and methods common for listening test datasets.
package data

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/dgryski/go-onlinestats"
	"github.com/google/zimtohrli/go/aio"
	"github.com/google/zimtohrli/go/audio"
	"github.com/google/zimtohrli/go/worker"

	_ "github.com/mattn/go-sqlite3" // To open sqlite3-databases.
)

const (
	// MOS is mean opinion score from human evaluators.
	MOS ScoreType = "MOS"
	// Zimtohrli is the Zimtohrli distance.
	Zimtohrli ScoreType = "Zimtohrli"
	// JND is 1 if the evaluator detected a difference and 0 if not.
	JND ScoreType = "JND"
	// ViSQOL is the ViSQOL MOS.
	ViSQOL = "ViSQOL"
)

// ScoreType represents a type of score, such as MOS or Zimtohrli.
type ScoreType string

// Better returns 1 if higher is better for the score type, or -1 if lower is better.
func (s ScoreType) Better() int {
	switch s {
	case MOS:
		return 1
	case Zimtohrli:
		return -1
	case JND:
		return -1
	case ViSQOL:
		return 1
	default:
		return 0
	}
}

// ScoreTypes is a slice of ScoreType.
type ScoreTypes []ScoreType

func (s ScoreTypes) Len() int {
	return len(s)
}

func (s ScoreTypes) Less(i, j int) bool {
	return string(s[i]) < string(s[j])
}

func (s ScoreTypes) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}

// Study contains data from a study.
type Study struct {
	dir string
	db  *sql.DB
}

// OpenStudy opens a study from a database directory.
// If the study doesn't exist, it will be created.
func OpenStudy(dir string) (*Study, error) {
	err := os.MkdirAll(dir, 0755)
	if err != nil && !os.IsExist(err) {
		return nil, fmt.Errorf("trying to create %q: %v", dir, err)
	}
	dbPath := filepath.Join(dir, "db.sqlite3")
	_, err = os.Stat(dbPath)
	if os.IsNotExist(err) {
		if _, err = os.Create(dbPath); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, fmt.Errorf("trying to open %q: %v", dbPath, err)
	}
	if _, err := db.Exec("CREATE TABLE IF NOT EXISTS OBJ (ID BLOB PRIMARY KEY, DATA BLOB)"); err != nil {
		return nil, fmt.Errorf("trying to ensure object table: %v", err)
	}
	return &Study{
		dir: dir,
		db:  db,
	}, nil
}

// Close closes the study.
func (s *Study) Close() error {
	return s.db.Close()
}

// AccuracyScore contains the accuracy for a metric when used to predict audible differences, and the threshold when that accuracy was achieved.
type AccuracyScore struct {
	ScoreType ScoreType
	Threshold float64
	Accuracy  float64
}

// AccuracyScores contains the accuracy scores for multiple score types.
type AccuracyScores []AccuracyScore

func (a AccuracyScores) String() string {
	table := Table{Row{"Score type", "Threshold", "Accuracy"}}
	for _, score := range a {
		table = append(table, Row{string(score.ScoreType), fmt.Sprintf("%.2v", score.Threshold), fmt.Sprintf("%.2f", score.Accuracy)})
	}
	return fmt.Sprintf("Maximal audibility classification threshold and accuracy per score type\n%s", table.String(2))
}

func abs(i int) int {
	if i < 0 {
		return -1
	}
	return i
}

func ternarySearch(f func(int) float64, left, right int) int {
	for abs(right-left) > 2 {
		third := (right - left) / 3
		leftThird := left + third
		rightThird := right - third
		if f(leftThird) < f(rightThird) {
			left = leftThird
		} else {
			right = rightThird
		}
	}
	return (left + right) / 2
}

// Accuracy returns the accuracy of each score type when used to predict audible differences.
func (s *Study) Accuracy() (AccuracyScores, error) {
	audibleMap := map[ScoreType]sort.Float64Slice{}
	inaudibleMap := map[ScoreType]sort.Float64Slice{}
	allMapMap := map[ScoreType]map[float64]struct{}{}
	if err := s.ViewEachReference(func(ref *Reference) error {
		for _, dist := range ref.Distortions {
			jnd, found := dist.Scores[JND]
			if !found {
				return fmt.Errorf("%+v doesn't have a JND score", ref)
			}
			for scoreType, score := range dist.Scores {
				if scoreType == JND {
					continue
				}
				scoreAll, found := allMapMap[scoreType]
				if !found {
					scoreAll = map[float64]struct{}{}
					allMapMap[scoreType] = scoreAll
				}
				scoreAll[score] = struct{}{}
				switch jnd {
				case 0:
					inaudibleMap[scoreType] = append(inaudibleMap[scoreType], score)
				case 1:
					audibleMap[scoreType] = append(audibleMap[scoreType], score)
				default:
					return fmt.Errorf("%+v JND isn't 0 or 1", ref)
				}
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	result := AccuracyScores{}
	for scoreType := range allMapMap {
		audible := audibleMap[scoreType]
		inaudible := inaudibleMap[scoreType]
		sort.Sort(audible)
		sort.Sort(inaudible)
		all := sort.Float64Slice{}
		for score := range allMapMap[scoreType] {
			all = append(all, score)
		}
		sort.Sort(all)
		accuracy := func(index int) float64 {
			threshold := all[index]
			audibleBelowThreshold := sort.SearchFloat64s(audible, threshold)
			inaudibleBelowThreshold := sort.SearchFloat64s(inaudible, threshold)
			correctAudible, correctInaudible := 0, 0
			if scoreType.Better() > 0 {
				correctAudible = audibleBelowThreshold
				correctInaudible = len(inaudible) - inaudibleBelowThreshold
			} else {
				correctAudible = len(audible) - audibleBelowThreshold
				correctInaudible = inaudibleBelowThreshold
			}
			return float64(correctAudible+correctInaudible) / float64(len(audible)+len(inaudible))
		}
		bestAccuracyThresholdIndex := ternarySearch(accuracy, 0, len(all)-1)
		result = append(result, AccuracyScore{
			ScoreType: scoreType,
			Threshold: all[bestAccuracyThresholdIndex],
			Accuracy:  accuracy(bestAccuracyThresholdIndex),
		})
	}
	return result, nil
}

// CorrelationScore contains the scorrelation score between two score types.
type CorrelationScore struct {
	ScoreTypeA ScoreType
	ScoreTypeB ScoreType
	Score      float64
}

// CorrelationTable contains the pairwise correlations between a set of score types.
type CorrelationTable [][]CorrelationScore

func (c CorrelationTable) String() string {
	result := Table{}
	header := Row{""}
	for _, score := range c[0] {
		header = append(header, string(score.ScoreTypeB))
	}
	result = append(result, header)
	for _, scores := range c {
		row := Row{string(scores[0].ScoreTypeA)}
		for _, score := range scores {
			row = append(row, fmt.Sprintf("%.2f", score.Score))
		}
		result = append(result, row)
	}
	return result.String(2)
}

// Correlate returns a table of all scores in the study Spearman correlated to each other.
func (s *Study) Correlate() (CorrelationTable, error) {
	scores := map[ScoreType][]float64{}
	if err := s.ViewEachReference(func(ref *Reference) error {
		for _, dist := range ref.Distortions {
			for scoreType, score := range dist.Scores {
				scores[scoreType] = append(scores[scoreType], score)
			}
		}
		return nil
	}); err != nil {
		return nil, err
	}
	sortedScoreTypes := ScoreTypes{}
	for scoreType := range scores {
		// Can't correlate JND.
		if scoreType != JND {
			sortedScoreTypes = append(sortedScoreTypes, scoreType)
		}
	}
	sort.Sort(sortedScoreTypes)
	result := CorrelationTable{}
	for _, scoreTypeA := range sortedScoreTypes {
		row := []CorrelationScore{}
		for _, scoreTypeB := range sortedScoreTypes {
			spearman, _ := onlinestats.Spearman(scores[scoreTypeA], scores[scoreTypeB])
			row = append(row, CorrelationScore{
				ScoreTypeA: scoreTypeA,
				ScoreTypeB: scoreTypeB,
				Score:      math.Abs(spearman),
			})
		}
		result = append(result, row)
	}
	return result, nil
}

// Measurement returns distance between sounds.
type Measurement func(reference, distortion *audio.Audio) (float64, error)

// Calculate computes measurements and populates the scores of the distortions.
func (s *Study) Calculate(measurements map[ScoreType]Measurement, pool *worker.Pool[any]) error {
	refs := []*Reference{}
	if err := s.ViewEachReference(func(ref *Reference) error {
		refs = append(refs, ref)
		return nil
	}); err != nil {
		return err
	}
	for _, loopRef := range refs {
		ref := loopRef
		pool.Submit(func(func(any)) error {
			refAudio, err := ref.Load(s.dir)
			if err != nil {
				log.Fatal(err)
			}
			for _, loopDist := range ref.Distortions {
				dist := loopDist
				pool.Submit(func(func(any)) error {
					distAudio, err := dist.Load(s.dir)
					if err != nil {
						return err
					}
					for loopScoreType := range measurements {
						scoreType := loopScoreType
						pool.Submit(func(func(any)) error {
							score, err := measurements[scoreType](refAudio, distAudio)
							if err != nil {
								return err
							}
							dist.Scores[scoreType] = score
							return nil
						})
					}
					return nil
				})
			}
			return nil
		})
	}
	if err := pool.Error(); err != nil {
		log.Println(err.Error())
	}
	if err := s.Put(refs); err != nil {
		return err
	}
	return nil
}

// ViewEachReference returns each reference in the study.
func (s *Study) ViewEachReference(f func(*Reference) error) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	rows, err := tx.Query("SELECT DATA FROM OBJ")
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var value []byte
		if err := rows.Scan(&value); err != nil {
			return err
		}
		ref := &Reference{}
		if err := json.Unmarshal(value, ref); err != nil {
			return err
		}
		if err := f(ref); err == io.EOF {
			break
		} else if err != nil {
			return err
		}
	}
	return nil
}

// Put inserts some references into a study.
func (s *Study) Put(refs []*Reference) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	if err := func() error {
		for _, ref := range refs {
			b, err := json.Marshal(ref)
			if err != nil {
				return err
			}
			if _, err = tx.Exec("INSERT INTO OBJ (ID, DATA) VALUES (?, ?) ON CONFLICT (ID) DO UPDATE SET DATA = ?", []byte(ref.Name), b, b); err != nil {
				return err
			}
		}
		return nil
	}(); err != nil {
		if rerr := tx.Rollback(); rerr != nil {
			return rerr
		}
		return err
	}
	return tx.Commit()
}

// Distortion contains data for a distortion of a reference.
type Distortion struct {
	Name   string
	Path   string
	Scores map[ScoreType]float64
}

// Load returns the audio for this distortion.
func (d *Distortion) Load(dir string) (*audio.Audio, error) {
	return aio.Load(filepath.Join(dir, d.Path))
}

// Reference contains data for a reference.
type Reference struct {
	Name        string
	Path        string
	Distortions []*Distortion
}

// Load returns the audio for this reference.
func (r *Reference) Load(dir string) (*audio.Audio, error) {
	return aio.Load(filepath.Join(dir, r.Path))
}
