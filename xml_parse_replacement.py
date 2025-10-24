    def _parse_verifiable_user_questions(self, response: str) -> List[VerifiableUserQuestion]:
        """Parse LLM response with XML tags into VerifiableUserQuestion objects"""
        import re
        
        print(f"  Parsing verifiable questions response (length: {len(response)})")
        questions = []
        
        # Extract all question blocks using regex
        question_pattern = r'<question>(.*?)</question>'
        question_blocks = re.findall(question_pattern, response, re.DOTALL)
        
        print(f"  Found {len(question_blocks)} question blocks")
        
        for i, block in enumerate(question_blocks):
            try:
                # Extract individual fields using regex
                type_match = re.search(r'<type>(.*?)</type>', block, re.DOTALL)
                level_match = re.search(r'<level>(.*?)</level>', block, re.DOTALL)
                verification_match = re.search(r'<verification>(.*?)</verification>', block, re.DOTALL)
                text_match = re.search(r'<text>(.*?)</text>', block, re.DOTALL)
                format_match = re.search(r'<format>(.*?)</format>', block, re.DOTALL)
                criteria_match = re.search(r'<criteria>(.*?)</criteria>', block, re.DOTALL)
                
                # Validate all fields are present
                if not all([type_match, level_match, text_match, format_match, criteria_match]):
                    print(f"  Question {i+1}: Missing required fields, skipping")
                    continue
                    
                # Extract and clean field values
                question_type_str = type_match.group(1).strip().upper()
                level = level_match.group(1).strip().lower()
                verification = verification_match.group(1).strip() if verification_match else ""
                question_text = text_match.group(1).strip()
                expected_format = format_match.group(1).strip()
                criteria = criteria_match.group(1).strip()
                
                # Validate question text is substantial
                if not question_text or len(question_text) < 10:
                    print(f"  Question {i+1}: Question text too short, skipping")
                    continue
                    
                # Map type to verification method and question type
                if question_type_str == 'NUMERICAL':
                    verification_method = VerificationMethod.NUMERICAL
                    question_type = 'quantitative_analysis'
                elif question_type_str == 'MATHEMATICAL':
                    verification_method = VerificationMethod.MATHEMATICAL
                    question_type = 'calculation_based'
                elif question_type_str == 'CLASSIFICATION':
                    verification_method = VerificationMethod.CATEGORICAL
                    question_type = 'classification'
                elif question_type_str == 'IDENTIFICATION':
                    verification_method = VerificationMethod.SINGLE_WORD
                    question_type = 'identification'
                elif question_type_str == 'CODE':
                    verification_method = VerificationMethod.CODE
                    question_type = 'code_generation'
                else:
                    print(f"  Question {i+1}: Unknown type '{question_type_str}', skipping")
                    continue
                
                # Create question object
                questions.append(VerifiableUserQuestion(
                    question=question_text,
                    question_type=question_type,
                    difficulty_level=level,
                    target_audience='researcher',
                    verification_method=verification_method,
                    expected_answer_format=expected_format,
                    verification_criteria=criteria
                ))
                
                print(f"  Question {i+1}: Successfully parsed ({question_type_str}): {question_text[:50]}...")
                
            except Exception as e:
                print(f"  Question {i+1}: Failed to parse - {e}")
                continue
        
        print(f"  Successfully parsed {len(questions)} verifiable questions")
        
        if len(questions) == 0:
            raise ValueError("Failed to parse any valid questions from LLM response. The response format may be incorrect or the questions may not meet quality standards.")
        
        # Filter for numerical/mathematical questions only
        priority_questions = [q for q in questions if q.verification_method in [VerificationMethod.NUMERICAL, VerificationMethod.MATHEMATICAL]]
        if len(priority_questions) == 0:
            raise ValueError(f"No NUMERICAL or MATHEMATICAL questions found among {len(questions)} parsed questions. All generated questions must be of these types for researcher-level verification.")
        
        return questions[:8]  # Limit to 8 questions