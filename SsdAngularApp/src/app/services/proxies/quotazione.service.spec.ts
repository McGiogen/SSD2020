import { TestBed } from '@angular/core/testing';

import { QuotazioneService } from './quotazione.service';

describe('QuotazioneService', () => {
  let service: QuotazioneService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(QuotazioneService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
