import { ChangeEvent, FocusEvent } from "react"

export interface NumberInputProps {
    label: string
    value: number
    name?: string
    onChange: (event: ChangeEvent) => void
    onBlur?: (event: FocusEvent) => void
}

export const NumberInput = (props: NumberInputProps) => {
    return (
        <div>
            <label>{props.label}</label>
            <input
                name={props.name}
                value={props.value}
                onChange={props.onChange}
                onBlur={props.onBlur}
                />
        </div>
    )
}