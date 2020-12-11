import { Component, OnInit } from '@angular/core';
import { FormControl, FormGroup } from '@angular/forms';
import { BehaviorSubject } from 'rxjs';
import { Quotazione } from '../model/quotazione';
import { QuotazioneService } from '../services/proxies/quotazione.service';

@Component({
  selector: 'app-rest-form',
  templateUrl: './rest-form.component.html',
  styleUrls: ['./rest-form.component.css']
})
export class RestFormComponent implements OnInit {

  latestResult$ = new BehaviorSubject<unknown>(null);

  form = new FormGroup({
    id: new FormControl(1),
    data: new FormControl(this.formatDate()),
    SP_500: new FormControl(),
    FTSE_MIB: new FormControl(),
    GOLD_SPOT: new FormControl(),
    MSCI_EM: new FormControl(),
    MSCI_EURO: new FormControl(),
    All_Bonds: new FormControl(),
    US_Treasury: new FormControl(),
  });

  constructor(
    private quotazioneService: QuotazioneService,
  ) { }

  ngOnInit(): void {
  }

  findAll(): void {
    this.quotazioneService.list()
      .subscribe(result => this.latestResult$.next(result));
  }

  findById(): void {
    const id = this.getFormValue().id;
    this.quotazioneService.get(id)
      .subscribe(result => this.latestResult$.next(result));
  }

  create(): void {
    const { id, ...data } = this.getFormValue();
    this.quotazioneService.create(data)
      .subscribe(result => this.latestResult$.next(result));
  }

  updateById(): void {
    const data = this.getFormValue();
    this.quotazioneService.update(data)
      .subscribe(result => this.latestResult$.next(result));
  }

  deleteById(): void {
    const id = this.getFormValue().id;
    this.quotazioneService.delete(id)
      .subscribe(result => this.latestResult$.next(result));
  }

  private getFormValue(): Quotazione {
    const formValue = this.form.value;
    formValue.data = this.formatDate(formValue.data);
    return formValue;
  }

  /**
   * Crea la data nel formato richiesto dal server, cioè M/D/YYYY
   * @param date Opzionale. Se non specificato, usa la data di oggi
   */
  private formatDate(date?: string): string {
    return (date == null ? new Date() : new Date(date)).toLocaleDateString('en-US');
  }
}
